import os

from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader

def main():

    load_dotenv()
    ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")

    loader = AzureBlobStorageLoader(
        account_url=ACCOUNT_URL,
        container_name="testcontainer",
        credential=InteractiveBrowserCredential() # TODO: go back to key-based auth
        )

    # test what lazy load returns!
    for doc in loader.lazy_load():
        print(doc.page_content)
    
    # next step: consider full doc loading in RAG pipeline...

if __name__ == '__main__':
    main()