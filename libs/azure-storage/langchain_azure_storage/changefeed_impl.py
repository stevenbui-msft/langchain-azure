'''
Feed in a time (consider the all the events AFTER that time) 

Parse the changefeed log based on that time 
	- parse the “file path” 
		- get the avro file to parse for (eventType, subject_blob) 

Output: list of (eventType, subject_blob) 
	- consider “subject” as way to reference the blob for LangChain
'''

# import libraries
import os
from azure.storage.blob import ContainerClient
from avro import datafile, io
from io import BytesIO
from collections import defaultdict
from datetime import datetime, timedelta, timezone

CONN_STR = os.getenv('CONN_STR')
CHANGEFEED_CONTAINER = '$blobchangefeed'
PST = timezone(timedelta(hours=-8))


def parse_local_datetime(date_text, time_text):
    base_date = datetime.strptime(date_text, '%Y/%m/%d')

    if time_text == '24:00':
        parsed_dt = base_date + timedelta(days=1)
        return parsed_dt.replace(tzinfo=PST)

    parsed_time = datetime.strptime(time_text, '%H:%M')
    parsed_dt = base_date.replace(hour=parsed_time.hour, minute=parsed_time.minute)
    return parsed_dt.replace(tzinfo=PST)


def parse_event_time(event_time_text):
    return datetime.fromisoformat(event_time_text.replace('Z', '+00:00'))


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
    prefix_parts = cf_layout_version.split('/')
    blob_parts = blob_name.split('/')

    if blob_parts[:len(prefix_parts)] != prefix_parts:
        return None

    remaining_parts = blob_parts[len(prefix_parts):]
    if len(remaining_parts) != 5:
        return None

    year, month, day, utc_time, file_name = remaining_parts
    if file_name != '00000.avro':
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

    '''
    container_client = ContainerClient.from_connection_string(CONN_STR, 'testcontainer')
    for blob in container_client.list_blobs():
        print()
        print(blob)
        print()
    '''

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

    if end_utc_dt < start_utc_dt:
        raise ValueError('End datetime must be after start datetime.')

    # /log/00/ is the change feed storage layout version
    cf_layout_version = 'log/00'

    start_hour_utc = start_utc_dt.replace(minute=0, second=0, microsecond=0)
    end_hour_utc = end_utc_dt.replace(minute=0, second=0, microsecond=0)

    '''
    cf_events = [] # stores (event_type, subject_blob)
    event_type_blobs = defaultdict(list) # stores {event_type: [(blob.name, changefeed_avro_file), ...]}
    '''
    blobs_to_refresh = set()
    files_listed = 0
    files_scanned = 0
    # to get the blob path, consider https://<storage_account_name>.blob.core.windows.net/<container_name>/<blob_name>
    container_client = ContainerClient.from_connection_string(CONN_STR, CHANGEFEED_CONTAINER)

    '''
    for blob in container_client.list_blobs():
        print(blob.name)
    
    print('fu')
    '''

    print('Listing changefeed blobs only for relevant UTC date prefixes...')
    for date_prefix in iter_changefeed_date_prefixes(cf_layout_version, start_hour_utc, end_hour_utc):
        print(f'Checking prefix: {date_prefix}')
        for blob in container_client.list_blobs(name_starts_with=date_prefix):
            files_listed += 1

            # Fast skip for non-target files before datetime parsing
            if not blob.name.endswith('/00000.avro'):
                continue

            blob_hour_utc = parse_changefeed_blob_datetime(blob.name, cf_layout_version)
            if blob_hour_utc is None:
                continue
            if not (start_hour_utc <= blob_hour_utc <= end_hour_utc):
                continue
            
            avro_file = blob.name
            print(f'Processing changefeed file: {avro_file}')
            files_scanned += 1

            # download the changefeed Avro log file
            blob_client = container_client.get_blob_client(avro_file)
            avro_bytes = blob_client.download_blob().readall()
        
            # make avro stream
            avro_stream = BytesIO(avro_bytes)
            reader = datafile.DataFileReader(avro_stream, io.DatumReader())
            matched_events_this_file = 0
        
            # assume that subject stores the reference to the blob object (adjust later)
            for event in reader:
                event_time = parse_event_time(event['eventTime'])
                if not (start_utc_dt <= event_time <= end_utc_dt):
                    continue
                blobs_to_refresh.add(event['subject'])
                #event_type = event['eventType']
                #blob_subject = event['subject']
                #blobs_to_refresh.add((blob_subject, event_time, event_type, event['id'], avro_file))
                '''
                cf_events.append((event_type, blob_subject))
                event_type_blobs[event_type].append(blob_subject)
                '''
                matched_events_this_file += 1
            reader.close()

            #print(f'  {matched_events_this_file} events in this file. Total so far: {len(cf_events)}')
    '''
    event_number = 0
    for event in cf_events:
        print(event_number)
        print(event)
        print()
        event_number += 1
    

    print()
    print('Grouped by event type:')
    for event_type, blob_subjects in event_type_blobs.items():
        print(event_type, blob_subjects)
    '''
    #print(f'here are blobs that were modified between the given time: {blobs_to_refresh}')
    
    print()
    print(f'Blobs listed: {files_listed}')
    print(f'Files scanned: {files_scanned}')
    #print(f'Total events: {len(cf_events)}')

    for blob_path in blobs_to_refresh:
        print(blob_path)
        print()
    
    # look into how the LangChain side (DocLoader) handles diff event types to adjust this code

if __name__ == '__main__':
    main()