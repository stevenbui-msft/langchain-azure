"""Sample demonstrating a chatbot using Azure AI and Azure Search."""

import os
import warnings

from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from langchain_azure_ai.embeddings import AzureAIOpenAIApiEmbeddingsModel
from langchain_azure_ai.vectorstores import AzureSearch
from langchain_community.vectorstores.azuresearch import AzureSearchVectorStoreRetriever


load_dotenv()
warnings.filterwarnings("ignore", message=".*preview.*")

_AZURE_CREDENTIAL = DefaultAzureCredential()
_AI_CREDENTIAL = os.environ.get("AZURE_FOUNDRY_API_KEY") or _AZURE_CREDENTIAL
_SEARCH_KEY = os.environ.get("AZURE_AI_SEARCH_API_KEY")
_COGNITIVE_CREDENTIAL_SCOPES = {
    "credential_scopes": ["https://cognitiveservices.azure.com/.default"]
}


def get_chat_model() -> AzureAIOpenAIApiChatModel:
    """Initialize and return the Azure AI chat model."""
    chat_model = AzureAIOpenAIApiChatModel(
        endpoint=os.environ.get("AZURE_CHAT_ENDPOINT"),
        project_endpoint=os.environ.get("AZURE_PROJECT_ENDPOINT"),
        credential=_AI_CREDENTIAL,
        model=os.environ["AZURE_CHAT_MODEL"],
    )
    return chat_model


def get_azure_search() -> AzureSearch:
    """Initialize and return the Azure Search vector store."""
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
        index_name=os.environ.get("AZURE_AI_SEARCH_INDEX_NAME", "demo-documents"),
        embedding_function=embed_model,
        additional_search_client_options=_COGNITIVE_CREDENTIAL_SCOPES,
    )

    return azure_search


def create_retriever() -> AzureSearchVectorStoreRetriever:
    """Create and return a retriever from Azure Search."""
    azure_search = get_azure_search()
    retriever = azure_search.as_retriever(
        search_type="similarity",
        k=3,
    )
    return retriever


def get_response(
    query: str,
    retriever: AzureSearchVectorStoreRetriever,
    llm: AzureAIOpenAIApiChatModel,
) -> str:
    """Get a response from the LLM based on the retrieved documents."""
    documents = retriever.invoke(query)
    context = "\n\n".join(
        [f"Document {doc.metadata['source']}:\n{doc.page_content}" for doc in documents]
    )
    prompt = f"""You are an AI assistant. Use the following context to answer the
        question otherwise say you do not know. Include the URL for the document. 
        Documents: {context} Question: {query} Answer:"""
    response = llm.invoke(prompt)
    return response.content


def chatbot() -> None:
    """Main chatbot loop."""
    retriever = create_retriever()
    llm = get_chat_model()
    print(
        "Welcome! This chatbot answers questions based on indexed documents "
        "stored in a vector store. Press 'Enter' to quit."
    )

    while True:
        user_input = input("\nYou: ")
        if user_input == "":
            print("\nGoodbye!")
            break

        response = get_response(user_input, retriever, llm)
        print(f"\nAI: {response}")


if __name__ == "__main__":
    chatbot()
