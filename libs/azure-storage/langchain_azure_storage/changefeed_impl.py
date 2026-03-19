'''
Feed in a time range (YYYY/MM/DD and HH:MM) in PST to convert to UTC.

Parse the Azure changefeed log based on that time range:
	- parse file paths based on the time date prefixes
        - if they exist,
		    - get the 00000.avro file to parse the events for blobs_names in the file
                - the blobs listed are the ones that were modified during the timespan

Output: list of the modified blobs during the timespan
'''

import os
from azure.storage.blob import ContainerClient
from avro import datafile, io
from io import BytesIO
from datetime import datetime, timedelta, timezone

CONN_STR = os.getenv('CONN_STR')
CHANGEFEED_CONTAINER = '$blobchangefeed'
PST = timezone(timedelta(hours=-8))
SUBJECT_PREFIX = '/blobServices/default/containers/'

def parse_local_datetime(date_text, time_text):
    base_date = datetime.strptime(date_text, '%Y/%m/%d')

    if time_text == '24:00':
        parsed_dt = base_date + timedelta(days=1)
        return parsed_dt.replace(tzinfo=PST)

    parsed_time = datetime.strptime(time_text, '%H:%M')
    parsed_dt = base_date.replace(hour=parsed_time.hour, minute=parsed_time.minute)
    return parsed_dt.replace(tzinfo=PST)


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


def parse_changefeed_blob_datetime(blob_name, cf_layout_version):
    expected_prefix = f'{cf_layout_version}/'
    if not blob_name.startswith(expected_prefix):
        return None

    remaining_path = blob_name[len(expected_prefix):]
    remaining_parts = remaining_path.split('/')
    if len(remaining_parts) != 5:
        return None

    year, month, day, utc_time, file_name = remaining_parts
    if not file_name.endswith('.avro'):
        return None
    if len(utc_time) != 4 or not utc_time.isdigit():
        return None

    try:
        return datetime(
            int(year),
            int(month),
            int(day),
            int(utc_time[:2]),
            0,
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None

def main():

    if not CONN_STR:
        raise ValueError('CONN_STR is not set. Add it to .env and restart the terminal.')

    # start time
    start_date_text = input('Enter the start date (YYYY/MM/DD): ')
    start_time_text = input('Enter the start time (HH:MM) in PST: ')
    start_local_dt = parse_local_datetime(start_date_text, start_time_text)
    start_utc_dt = start_local_dt.astimezone(timezone.utc)

    # end time
    end_date_text = input('Enter the end date (YYYY/MM/DD): ')
    end_time_text = input('Enter the end time (HH:MM) in PST: ')
    end_local_dt = parse_local_datetime(end_date_text, end_time_text)
    end_utc_dt = end_local_dt.astimezone(timezone.utc)

    # TODO: could also support multiple containers (like a list of valid container names)
    container_filter = input('Enter container name to filter (leave blank for all containers): ').strip()

    if end_utc_dt < start_utc_dt:
        raise ValueError('End datetime must be after start datetime.')

    start_hour_utc = start_utc_dt.replace(minute=0, second=0, microsecond=0)
    end_hour_utc = end_utc_dt.replace(minute=0, second=0, microsecond=0)

    # /log/00/ is the change feed storage layout version
    cf_layout_version = 'log/00'
    container_client = ContainerClient.from_connection_string(CONN_STR, CHANGEFEED_CONTAINER)

    blobs_to_refresh = set()

    if container_filter:
        print(f'Filtering events from container: {container_filter}')
    else:
        print('No container filter set; including events from all containers.')

    print('Listing changefeed blobs only for relevant UTC date prefixes...')
    for date_prefix in iter_changefeed_date_prefixes(cf_layout_version, start_hour_utc, end_hour_utc):
        print(f'Checking prefix: {date_prefix}')
        for blob in container_client.list_blobs(name_starts_with=date_prefix):

            if not blob.name.endswith('.avro'):
                continue
            blob_hour_utc = parse_changefeed_blob_datetime(blob.name, cf_layout_version)
            if blob_hour_utc is None:
                continue
            # consider the hour valid range
            if not (start_hour_utc <= blob_hour_utc <= end_hour_utc):
                continue
            
            avro_file = blob.name
            print(f'Processing changefeed file: {avro_file}')

            # download the changefeed Avro log file
            blob_client = container_client.get_blob_client(avro_file)
            avro_bytes = blob_client.download_blob().readall()
        
            # make avro stream
            avro_stream = BytesIO(avro_bytes)
            reader = datafile.DataFileReader(avro_stream, io.DatumReader())
            
            # parse the avro file
            for event in reader:
                
                # eventType filtering (we only want BlobCreated and BlobPropertiesUpdated)
                eventType = event['eventType']
                if (eventType == 'BlobDeleted' or eventType == 'BlobSnapshotCreated' or
                    eventType == 'BlobAsyncOperationInitiated' or eventType == 'BlobTierChanged'):
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
                # if container filtering, check if this blob is even part of the container
                if container_filter and subject_blob_container_name != container_filter:
                    continue

                blobs_to_refresh.add(blob_name)

            reader.close()

    for blob_path in blobs_to_refresh:
        print(blob_path)
        print()
    
    # return list(blobs_to_refresh)

if __name__ == '__main__':
    main()