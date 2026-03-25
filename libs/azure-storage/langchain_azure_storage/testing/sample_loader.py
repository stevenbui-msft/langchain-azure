import os

from azure.identity import InteractiveBrowserCredential

from langchain_azure_storage.changefeed import main as changefeed_blobs_to_refresh
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

# python -m langchain_azure_storage.testing.sample_loader  

def main():
    # design: use the changefeed to fill blob_names with blobs to refresh
    blobs_to_refresh = changefeed_blobs_to_refresh()

    loader = AzureBlobStorageLoader(
        account_url=os.getenv('CONN_STR'),
        container_name="ACCOUNT_URL",
        blob_names=blobs_to_refresh,
        credential=InteractiveBrowserCredential(),
    )

    # test what is lazy laod returns!
    for doc in loader.lazy_load():
        print(doc.page_content)
    
    # next step: consider full doc loading in RAG pipeline...

if __name__ == '__main__':
    main()