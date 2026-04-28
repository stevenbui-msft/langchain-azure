# RAG Agent with AzureBlobStorageLoader Demo
This demo creates a RAG agent that responds to queries based on documents loaded from Azure Blob Storage.

## Quick Start

> **Note:** This demo requires configuring your environment to use [`DefaultAzureCredentials`](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) (e.g. [`Azure CLI`](https://learn.microsoft.com/en-us/cli/azure/authenticate-azure-cli?view=azure-cli-latest)). Azure AI Search also requires the `Search Index Data Contributor` and `Search Service Contributor` role assignments. To use `DefaultAzureCredentials`, you must enable [RBAC for your AI Search Service](https://learn.microsoft.com/en-us/azure/search/search-security-enable-roles?tabs=config-svc-portal%2Cdisable-keys-portal).

1. **Install dependencies:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   If PowerShell blocks activation scripts, you can run the virtual environment's Python directly as shown above.

2. **Configure environment variables:**

   Create a `.env` file and add values like these:

   ```dotenv
   AZURE_STORAGE_ACCOUNT_URL=https://<your-account-name>.blob.core.windows.net
   AZURE_STORAGE_CONTAINER_NAME=your-container-name
   AZURE_STORAGE_BLOB_PREFIX=

   AZURE_FOUNDRY_API_KEY=
   AZURE_AI_SEARCH_API_KEY=

   AZURE_EMBEDDING_MODEL=your-embedding-deployment-name
   AZURE_EMBEDDING_ENDPOINT=https://<your-openai-resource>.openai.azure.com/openai/v1

   AZURE_CHAT_MODEL=your-chat-deployment-name
   AZURE_CHAT_ENDPOINT=https://<your-openai-resource>.openai.azure.com/openai/v1

   AZURE_AI_SEARCH_ENDPOINT=https://<your-azure-search-resource-name>.search.windows.net
   AZURE_AI_SEARCH_INDEX_NAME=demo-documents
   ```

   If you have already signed in with Azure CLI or another supported tool, the API key fields can be left blank.

3. **Create vector store** (first time only):

   This step will list blobs as documents from an Azure Blob Storage container and save it to the Azure AI Search vector store. To specify which blobs to return, set the `AZURE_STORAGE_BLOB_PREFIX` environment variable, otherwise all blobs in the container will be returned.
   ```powershell
   .\.venv\Scripts\python.exe embed.py
   ```

4. **Run the agent:**

   This step runs the chatbot agent which uses the context saved to the Azure AI Search vector store to respond to questions.
   ```powershell
   .\.venv\Scripts\python.exe query.py
   ```

   **Sample interaction:**
   ```text
   You: What is Azure Blob Storage?

   AI: Azure Blob Storage is a service for storing large amounts of unstructured data...
   Source:  https://<your-account-name>.blob.core.windows.net/<your-container-name>/pdf_file.pdf
   ```