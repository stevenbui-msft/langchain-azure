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
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from avro import datafile, io
from io import BytesIO
from collections import defaultdict

CONN_STR = os.getenv('CONN_STR')
CHANGEFEED_CONTAINER = '$blobchangefeed'

# date time structure mm/dd/yy -> hour
#   consider 15 minute bound constraint

def main():
    if not CONN_STR:
        raise ValueError('CONN_STR is not set. Add it to .env and restart the terminal.')

    # read user input for time (for simplicity, ask for YYYY/MM/DD and time HH:MM in PST)
    # also add error handling later on
    inputted_date = input('Enter the date (YYYY/MM/DD): ')
    year, month, day = inputted_date.split('/')
    inputted_time = input('Enter the time (HH:MM) in PST: ')
    hour_str, minute_str = inputted_time.split(':')
    hour, minute = int(hour_str), int(minute_str)
    utc_hour = (hour + 8) % 24
    utc_time = f'{utc_hour:02d}{minute:02d}'

    # read the correct avro file after time
    # /log/00/  is the change feed storage layout version
    cf_layout_version = 'log/00'
    blob_file_path = '/'.join([cf_layout_version, year, month, day, utc_time, '00000.avro'])

    blob_client = BlobClient.from_connection_string(CONN_STR, CHANGEFEED_CONTAINER, blob_file_path)
    stuff = blob_client.download_blob().readall()
    
    # make avro stream
    avro_stream = BytesIO(stuff)
    reader = datafile.DataFileReader(avro_stream, io.DatumReader())
    
    # assume that subject stores the reference to the blob object
    cf_events = [] # stores (event_type_blobs, blob_subject) 
    event_type_blobs = defaultdict(list) # stores {event_type_blobs: blob_subject}
    for event in reader:
        event_type = event['eventType']
        blob_subject = event['subject']
        cf_events.append((event_type, blob_subject))
        event_type_blobs[event_type].append(blob_subject)
    reader.close()

    event_number = 0
    for event in cf_events:
        print(event_number)
        print(event)
        print()
        event_number += 1
    
    # look into how the LangChain side (DocLoader) handles diff event types to adjust this code

if __name__ == '__main__':
    main()