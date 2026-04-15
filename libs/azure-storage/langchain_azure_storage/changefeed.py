'''
Feed in a time range (YYYY/MM/DD and HH:MM) in PST to convert to UTC.

Parse the Azure changefeed log based on that time range:
	- parse file paths based on the time date prefixes
        - if they exist,
		    - get the 00000.avro file to parse the events for blobs_names in the file
                - the blobs listed are the ones that were modified during the timespan

Logs and prints list of blobs deleted during timespan (for future handling of blob_deletion in vector store)       

Output: list of the modified blobs during the timespan
'''

import os
from azure.storage.blob import ContainerClient
from avro import datafile, io
from io import BytesIO
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CONN_STR = os.getenv('CONN_STR')
CHANGEFEED_CONTAINER = '$blobchangefeed'
SUBJECT_PREFIX = '/blobServices/default/containers/'


def parse_local_datetime(date_text, time_text):
    local_date = datetime.strptime(date_text, '%Y/%m/%d')
    if time_text == '24:00':
        local_date = local_date + timedelta(days=1)
        naive_dt = local_date.replace(hour=0, minute=0)
    else:
        parsed_time = datetime.strptime(time_text, '%H:%M')
        naive_dt = local_date.replace(hour=parsed_time.hour, minute=parsed_time.minute)

    try:
        return naive_dt.replace(tzinfo=ZoneInfo('America/Los_Angeles'))
    except ZoneInfoNotFoundError:
        year = naive_dt.year
        march_first = datetime(year, 3, 1)
        november_first = datetime(year, 11, 1)
        second_sunday_march_day = 1 + ((6 - march_first.weekday()) % 7) + 7
        first_sunday_november_day = 1 + ((6 - november_first.weekday()) % 7)
        dst_start = datetime(year, 3, second_sunday_march_day, 2, 0)
        dst_end = datetime(year, 11, first_sunday_november_day, 2, 0)
        offset_hours = -7 if dst_start <= naive_dt < dst_end else -8
        return naive_dt.replace(tzinfo=timezone(timedelta(hours=offset_hours)))


def iter_changefeed_date_prefixes(cf_layout_version, start_utc_dt, end_utc_dt):
    current_date_utc = start_utc_dt.date()
    end_date_utc = end_utc_dt.date()

    while current_date_utc <= end_date_utc:
        yield '/'.join([
            cf_layout_version,
            f'{current_date_utc.year:04d}',
            f'{current_date_utc.month:02d}',
            f'{current_date_utc.day:02d}',
            '',
        ])
        current_date_utc += timedelta(days=1)


def main(container_name, start_date_text, start_time_text, end_date_text, end_time_text):

    if not CONN_STR:
        raise ValueError('CONN_STR is not set. Add it to .env and restart the terminal.')

    # start time
    start_local_dt = parse_local_datetime(start_date_text, start_time_text)
    start_utc_dt = start_local_dt.astimezone(timezone.utc)

    # end time
    end_local_dt = parse_local_datetime(end_date_text, end_time_text)
    end_utc_dt = end_local_dt.astimezone(timezone.utc)

    container_filter = container_name.strip().lower()

    if end_utc_dt < start_utc_dt:
        raise ValueError('End datetime must be after start datetime.')

    cf_layout_version = 'log/00'
    container_client = ContainerClient.from_connection_string(CONN_STR, CHANGEFEED_CONTAINER)

    blobs_to_refresh = set()
    blobs_deleted = set()

    print('Listing changefeed blobs for UTC date prefixes...')
    for date_prefix in iter_changefeed_date_prefixes(cf_layout_version, start_utc_dt, end_utc_dt):
        for blob in container_client.list_blobs(name_starts_with=date_prefix):
            if not blob.name.endswith('.avro'):
                continue

            # download the changefeed Avro log file     
            blob_client = container_client.get_blob_client(blob.name)
            avro_bytes = blob_client.download_blob().readall()

            # make avro stream
            avro_stream = BytesIO(avro_bytes)
            reader = datafile.DataFileReader(avro_stream, io.DatumReader())

            # parse the avro file
            for event in reader:
                eventType = event['eventType']
                if eventType == 'BlobAsyncOperationInitiated':
                    continue
                if not (eventType == 'BlobCreated' or eventType == 'BlobPropertiesUpdated' or eventType == 'BlobDeleted'):
                    continue

                event_time = datetime.fromisoformat(event['eventTime'].replace('Z', '+00:00'))
                # consider the minute valid range
                if not (start_utc_dt <= event_time <= end_utc_dt):
                    continue

                subject_blob_path = event['subject']
                # sample path: /blobServices/default/containers/testcontainer/blobs/more2
                # path in the format of /blobServices/default/containers/{CONTAINER_NAME}/blobs/{BLOB_NAME}
                if not subject_blob_path.startswith(SUBJECT_PREFIX):
                    continue

                container_and_blob = subject_blob_path[len(SUBJECT_PREFIX):]
                split_parts = container_and_blob.split('/blobs/', 1)
                if len(split_parts) != 2:
                    continue

                subject_blob_container_name, blob_name = split_parts
                if not subject_blob_container_name or not blob_name:
                    continue
                if container_filter and subject_blob_container_name.lower() != container_filter:
                    continue

                # remove deleted blobs if it's in the list of blob to refresh
                # otherwise, just log it to the blobs_deleted set
                if eventType == 'BlobDeleted':
                    if blob_name in blobs_to_refresh:
                        blobs_to_refresh.remove(blob_name)
                    else:
                        blobs_deleted.add(blob_name)
                    continue

                blobs_to_refresh.add(blob_name)

            reader.close()

    print("===== BLOBS DELETED =====")
    for blob_name in blobs_deleted:
        print(blob_name)

    return blobs_to_refresh

if __name__ == '__main__':
    main()