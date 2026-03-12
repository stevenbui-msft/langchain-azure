import os
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient, changefeed
from avro import datafile, io
from io import BytesIO

CONN_STR = os.getenv('CONN_STR')
CHANGEFEED_CONTAINER = '$blobchangefeed'

def main():
    if not CONN_STR:
        raise ValueError('CONN_STR is not set. Add it to .env and restart the terminal.')
    
    blob_client = BlobClient.from_connection_string(CONN_STR, CHANGEFEED_CONTAINER, 'log/00/2026/02/25/1900/00000.avro')
    stuff = blob_client.download_blob().readall()

    
    # make avro stream
    avro_stream = BytesIO(stuff)
    reader = datafile.DataFileReader(avro_stream, io.DatumReader())
    # dict_keys(['schemaVersion', 'topic', 'subject', 'eventType', 'eventTime', 'id', 'data'])
    # we notice that:
        # we always use schemaVersion: 6
        # topic: account details
        # subject: the blob we do stuff with (!!!!! could be useful to index which blobs actaully changed so we can doc reload it !!!!!)
        # eventType: deleted, created, etc (!!! key ID to explore !!!)
        # eventTime: example -> '2026-02-25T19:32:00.7684389Z' (good when we want to do TIME operations)
        # id: id of the cf log
        # data: idk (could just consider an abstraction of this)
    
    # Change event records where the eventType has a value of Control are internal system records and don't reflect a change to objects in your account.
    # You can safely ignore those records.

    # the time represented by the segment is approximate with bounds of 15 minutes. So to ensure consumption of all records within a specified time,
    # consume the consecutive previous and next hour segment.

    for event in reader:
        print(event['eventType'])
        print()
    reader.close()
    '''

    # use changefeed
    cf_client = changefeed.ChangeFeedClient.from_connection_string(CONN_STR)
    for change in cf_client.list_changes():
        print(change.get('eventType'))
        print()
        print()
    '''


if __name__ == '__main__':
    main()
