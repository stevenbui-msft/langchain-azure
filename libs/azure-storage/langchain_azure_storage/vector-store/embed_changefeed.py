"""Sample showing embedding documents from Azure Blob Storage into Azure Search."""

import logging
import os
import warnings

from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel
from langchain_azure_ai.vectorstores import AzureSearch
from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

logger = logging.getLogger("pypdf")
logger.setLevel(logging.ERROR)

load_dotenv()
warnings.filterwarnings("ignore", message=".*preview.*")

_AZURE_CREDENTIAL = DefaultAzureCredential()
_AI_CREDENTIAL = os.environ.get("AZURE_FOUNDRY_API_KEY") or _AZURE_CREDENTIAL
_SEARCH_KEY = os.environ.get("AZURE_AI_SEARCH_API_KEY")
_COGNITIVE_CREDENTIAL_SCOPES = {
    "credential_scopes": ["https://cognitiveservices.azure.com/.default"]
}
_EMBED_BATCH_SIZE = 50


def main() -> None:
    """Embed documents from Azure Blob Storage into Azure Search."""
    preview_count = int(os.environ.get("VECTOR_STORE_PREVIEW_COUNT", "10"))
    inspect_only = os.environ.get("INSPECT_ONLY", "0") == "1"

    if inspect_only:
        print("Running in INSPECT_ONLY mode: no blob loading or embeddings will be performed.")
        print_search_index_contents(
            get_search_client(),
            limit=preview_count,
            title="Inspect-only mode",
        )
        return

    print("Running in INDEXING mode: blobs will be loaded and embeddings will be generated.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=12000,
        chunk_overlap=500,
    )

    embed_model = AzureAIOpenAIApiEmbeddingsModel(
        endpoint=os.environ.get("AZURE_EMBEDDING_ENDPOINT"),
        project_endpoint=os.environ.get("AZURE_PROJECT_ENDPOINT"),
        credential=_AI_CREDENTIAL,
        model=os.environ["AZURE_EMBEDDING_MODEL"],
    )

    azure_search = AzureSearch(
        azure_search_endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        azure_search_key=_SEARCH_KEY,
        azure_credential=None if _SEARCH_KEY else _AZURE_CREDENTIAL,
        additional_search_client_options=_COGNITIVE_CREDENTIAL_SCOPES,
        index_name=os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", "demo-documents"),
        embedding_function=embed_model,
    )

    loader = AzureBlobStorageLoader(
        account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        container_name=os.environ["AZURE_STORAGE_CONTAINER_NAME"],
        prefix=os.environ.get("AZURE_STORAGE_BLOB_PREFIX"),
        credential=_AZURE_CREDENTIAL,
        loader_factory=PyPDFLoader,
    )

    docs = []
    total_processed = 0
    blobs_seen = set()
    blob_progress = get_progress_bar()
    preview_count = int(os.environ.get("VECTOR_STORE_PREVIEW_COUNT", "10"))

    if os.environ.get("PRINT_VECTOR_STORE_BEFORE_INDEXING", "0") == "1":
        print_search_index_contents(
            get_search_client(),
            limit=preview_count,
            title="Before indexing",
        )

    for doc in loader.lazy_load():
        update_progress_bar(doc, blobs_seen, blob_progress)
        docs.extend(text_splitter.split_documents([doc]))

        if len(docs) >= _EMBED_BATCH_SIZE:
            azure_search.add_documents(docs)
            total_processed += len(docs)
            docs = []

    if docs:
        azure_search.add_documents(docs)
        total_processed += len(docs)

    blob_progress.close()
    print(
        f"Complete: {total_processed} documents across {len(blobs_seen)} blobs embedded and added to Azure Search index."
    )
    print_search_index_contents(get_search_client(), limit=preview_count, title="After indexing")


def get_search_client() -> SearchClient:
    """Return a search client for direct index inspection."""
    credential = AzureKeyCredential(_SEARCH_KEY) if _SEARCH_KEY else _AZURE_CREDENTIAL
    return SearchClient(
        endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        index_name=os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", "demo-documents"),
        credential=credential,
    )


def print_search_index_contents(
    search_client: SearchClient, limit: int = 10, title: str = "Vector store preview"
) -> None:
    """Print a preview of the stored chunks from Azure Search."""
    print(f"\n{title} ({limit} chunks max):")
    results = search_client.search(
        search_text="*",
        top=limit,
        select=["id", "content", "metadata"],
    )

    for index, item in enumerate(results, start=1):
        content = (item.get("content") or "").replace("\n", " ")
        print(f"\n[{index}] id={item.get('id')}")
        print(f"metadata={item.get('metadata')}")
        print(f"content={content[:200]}")



def get_progress_bar() -> tqdm:
    blob_service_client = BlobServiceClient(
        account_url=os.environ["AZURE_STORAGE_ACCOUNT_URL"],
        credential=_AZURE_CREDENTIAL,
    )
    container_client = blob_service_client.get_container_client(
        os.environ["AZURE_STORAGE_CONTAINER_NAME"]
    )
    prefix = os.environ.get("AZURE_STORAGE_BLOB_PREFIX")
    blob_list = list(container_client.list_blobs(name_starts_with=prefix))
    return tqdm(total=len(blob_list), desc="Processing blobs", unit=" blobs")


def update_progress_bar(doc, blobs_seen, blob_progress) -> None:
    blob_name = doc.metadata.get("source")
    if blob_name not in blobs_seen:
        blobs_seen.add(blob_name)
        blob_progress.update(1)


if __name__ == "__main__":
    main()
