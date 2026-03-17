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

def main():

    container_client = ContainerClient.from_connection_string(CONN_STR, CHANGEFEED_CONTAINER)
    for blob in container_client.list_blobs():
        print()
        print(blob)
        print()
    

if __name__ == '__main__':
    main()