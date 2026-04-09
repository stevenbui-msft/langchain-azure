import os

from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

#from langchain_azure_storage.changefeed import main as changefeed_blobs_to_refresh
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

# python -m langchain_azure_storage.testing.sample_loader  

def main():

    load_dotenv()
    ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")

    '''
    # design: use the changefeed to fill blob_names with blobs to refresh
    blobs_to_refresh = changefeed_blobs_to_refresh()

    loader = AzureBlobStorageLoader(
        account_url=ACCOUNT_URL,
        container_name="testcontainer",
        blob_names=blobs_to_refresh,
        credential=InteractiveBrowserCredential())
    '''

    loader = AzureBlobStorageLoader(
        account_url=ACCOUNT_URL,
        container_name="testcontainer",
        credential=InteractiveBrowserCredential(),
        start_date='2026/04/07',
        start_time='00:00',
        end_date='2026/04/07',
        end_time='23:59')

    # test what lazy load returns!
    for doc in loader.lazy_load():
        print(doc.page_content)
    
    # next step: consider full doc loading in RAG pipeline...

if __name__ == '__main__':
    main()