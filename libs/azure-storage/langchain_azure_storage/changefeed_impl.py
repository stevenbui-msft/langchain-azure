'''
Feed in a time (consider the all the events AFTER that time) 

Parse the changefeed log based on that time 
	- parse the “file path” 
		- get the avro file to parse for (eventType, subject) 

Output: list of (eventType, subject) 
	- consider “subject” as way to reference the blob for LangChain
'''

# import libraries
import os
from azure.storage.blob import BlobClient
from azure.core.exceptions import ResourceNotFoundError
from avro import datafile, io
from io import BytesIO
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from time import perf_counter

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


def build_changefeed_blob_path(cf_layout_version, utc_dt):
    hour_segment = utc_dt.strftime('%H00')
    return '/'.join([
        cf_layout_version,
        utc_dt.strftime('%Y'),
        utc_dt.strftime('%m'),
        utc_dt.strftime('%d'),
        hour_segment,
        '00000.avro',
    ])


def parse_event_time(event_time_text):
    return datetime.fromisoformat(event_time_text.replace('Z', '+00:00'))

def main():
    if not CONN_STR:
        raise ValueError('CONN_STR is not set. Add it to .env and restart the terminal.')

    started_at = perf_counter()

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

    # read the correct avro file after time for ALL files between start and end time
    # /log/00/  is the change feed storage layout version
    cf_layout_version = 'log/00'

    # we need a way to loop thorugh all the times,
    # i think oging by hour should work
    # so like we can floor the initial time to the hour and ceil the ending time the horu
    # and then loop thorugh each hour while a log with that hour exists
    # but then again, we run into the issue of a timespan spanning days maybe even months lol
    # ok, i will limit the scope irght now to the same day
    # or, i guess i can just apply the same prinicple, adding to the hour vars, and if curr_time is like a new day, consider day += 1
    # there also spans another questiona bout the time format lol. i was prev thinking 1-12 and not 1-24
    current_hour_utc = start_utc_dt.replace(minute=0, second=0, microsecond=0)
    end_hour_utc = end_utc_dt.replace(minute=0, second=0, microsecond=0)

    cf_events = [] # stores (event_type, blob_subject)
    event_type_blobs = defaultdict(list) # stores {event_type: [blob_subject, ...]}
    hours_checked = 0

    while current_hour_utc <= end_hour_utc:
        blob_file_path = build_changefeed_blob_path(cf_layout_version, current_hour_utc)
        print(f'Checking changefeed file: {blob_file_path}')
        hours_checked += 1

        blob_client = BlobClient.from_connection_string(CONN_STR, CHANGEFEED_CONTAINER, blob_file_path)
        try:
            stuff = blob_client.download_blob().readall()
        except ResourceNotFoundError:
            print('  Not found, skipping this hour.')
            current_hour_utc += timedelta(hours=1)
            continue

        print(f'  Downloaded {len(stuff)} bytes.')
        
        # make avro stream
        avro_stream = BytesIO(stuff)
        reader = datafile.DataFileReader(avro_stream, io.DatumReader())
        matched_events_this_file = 0
        
        # assume that subject stores the reference to the blob object
        for event in reader:
            event_time = parse_event_time(event['eventTime'])
            if not (start_utc_dt <= event_time <= end_utc_dt):
                continue
            event_type = event['eventType']
            blob_subject = event['subject']
            cf_events.append((event_type, blob_subject))
            event_type_blobs[event_type].append(blob_subject)
            matched_events_this_file += 1
        reader.close()

        print(f'  Matched {matched_events_this_file} events in this file. Total so far: {len(cf_events)}')

        current_hour_utc += timedelta(hours=1)

    event_number = 0
    for event in cf_events:
        print(event_number)
        print(event)
        print()
        event_number += 1

    print('Grouped by event type:')
    for event_type, blob_subjects in event_type_blobs.items():
        print(event_type, blob_subjects)

    elapsed_seconds = perf_counter() - started_at
    print()
    print(f'Completed scan in {elapsed_seconds:.2f} seconds.')
    print(f'Hours checked: {hours_checked}')
    print(f'Total matching events: {len(cf_events)}')
    
    # look into how the LangChain side (DocLoader) handles diff event types to adjust this code

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user.')