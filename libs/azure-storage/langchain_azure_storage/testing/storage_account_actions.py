import os
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient, changefeed
from avro import datafile, io
from io import BytesIO

CONN_STR = os.getenv('CONN_STR')
CHANGEFEED_LOG = '$blobchangefeed'

def main():
    if not CONN_STR:
        raise ValueError('CONN_STR is not set. Add it to .env and restart the terminal.')

    container_client = ContainerClient.from_connection_string(CONN_STR, 'testcontainer')
    container_client.create_container() if not container_client.exists() else None

    '''
    # TESTING

    # create a blob
    container_client.get_blob_client('make1').upload_blob('i create blob.') if not container_client.get_blob_client('make1').exists() else None

    # delete a blob
    container_client.get_blob_client('make1').delete_blob()

    # change properties of a blob
    container_client.get_blob_client('make1').upload_blob('new data test!', overwrite=True)
    '''

    '''
    # BLOB ACTIONS

    #container_client.get_blob_client('more1').set_blob_tags({"status": "processed", "source": "pipelineA"})
    #container_client.get_blob_client('more2').upload_blob('new data test!', overwrite=True)

    container_client.get_blob_client('afterhour1').delete_blob()

    container_client.get_blob_client('afterhour1').upload_blob('steven') if not container_client.get_blob_client('afterhour1').exists() else None
    container_client.get_blob_client('afterhour2').upload_blob('is') if not container_client.get_blob_client('afterhour2').exists() else None
    container_client.get_blob_client('afterhour3').upload_blob('number') if not container_client.get_blob_client('afterhour3').exists() else None
    container_client.get_blob_client('afterhour4').upload_blob(b'byte1') if not container_client.get_blob_client('afterhour4').exists() else None
    container_client.get_blob_client('afterhour5').upload_blob(b'byte2') if not container_client.get_blob_client('afterhour5').exists() else None

    container_client.get_blob_client('microsoft').delete_blob() if container_client.get_blob_client('microsoft').exists() else None
    container_client.get_blob_client('blob2').delete_blob() if container_client.get_blob_client('blob2').exists() else None
    '''
    

    '''
    # PRINT EVENTS

    blob_client = BlobClient.from_connection_string(CONN_STR, CHANGEFEED_LOG, 'log/00/2026/03/23/1700/00000.avro')
    stuff = blob_client.download_blob().readall()

    # make avro stream
    avro_stream = BytesIO(stuff)
    reader = datafile.DataFileReader(avro_stream, io.DatumReader())
    for event in reader:
        print(event)
    reader.close()
    '''
    blob_client = BlobClient.from_connection_string(CONN_STR, CHANGEFEED_LOG, 'log/00/2026/04/07/1600/00000.avro')
    stuff = blob_client.download_blob().readall()

    # make avro stream
    avro_stream = BytesIO(stuff)
    reader = datafile.DataFileReader(avro_stream, io.DatumReader())
    for event in reader:
        print(event)
        print()
    reader.close()

if __name__ == '__main__':
    main()