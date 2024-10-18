# RAG Personal Assistant

This tool leverages **Retrieval-Augmented Generation (RAG)** to answer user questions based on documents stored in a vector database. By combining **LangChain**'s document processing pipeline with **OpenAI's GPT-4** language model, this app offers precise, contextually rich responses to questions by retrieving relevant information from user-provided documents.

## Table of Contents

- [What is Retrieval-Augmented Generation (RAG)?](#what-is-retrieval-augmented-generation-rag)
- [Problem RAG Solves in Generative AI](#problem-rag-solves-in-generative-ai)
- [How This App Works](#how-this-app-works)
  - [Document Loading](#document-loading)
  - [Document Splitting](#document-splitting)
  - [Embedding and Vector Store](#embedding-and-vector-store)
  - [Retrieval and QA Chain](#retrieval-and-qa-chain)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [Future Improvements](#future-improvements)

---

## What is Retrieval-Augmented Generation (RAG)?

RAG is a hybrid technique that combines **information retrieval** and **generative AI** models to improve the accuracy and relevance of AI-generated responses. Instead of relying solely on a language model’s pre-trained knowledge, RAG first retrieves relevant documents from a dataset (or knowledge base) and then uses a generative model (like GPT-4) to generate answers using the retrieved documents as context.

In essence, RAG ensures that the AI has access to the **most up-to-date and specific** information for each query.

## Problem RAG Solves in Generative AI

One of the main challenges with generative AI (like GPT-4) is that it has a **fixed knowledge cutoff**. This means it can’t access information beyond what it was trained on. Relying solely on such models for tasks requiring **current** or **domain-specific knowledge** can lead to inaccuracies.

### RAG Solves This By:
1. **Connecting the Model to External Knowledge**: By retrieving relevant documents from a specific dataset, RAG allows the model to work with real-time and domain-specific information.
2. **Improving Accuracy**: The generative model provides answers using the retrieved documents, ensuring that responses are grounded in actual data.
3. **Reducing Hallucination**: Generative models often "hallucinate" when they don’t know the answer. By using RAG, the AI refers to actual documents, minimizing such errors.

## How This App Works

This app uses a RAG-based approach to create a personal knowledge assistant that can:
- Load and process documents from your local storage
- Split documents into smaller chunks for efficient embedding and retrieval
- Embed document chunks using **OpenAIEmbeddings**
- Store embeddings in a **Chroma** vector database with persistence
- Retrieve relevant documents based on user queries and generate answers using **OpenAI's GPT-4**

### Document Loading

Documents from various formats—**Text (.txt)**, **PDF (.pdf)**, and **Word (.docx)**—are loaded using **LangChain**'s `DirectoryLoader`. These documents are processed in a recursive manner, allowing for folder-by-folder document ingestion.

### Document Splitting

Long documents are split into smaller chunks using **RecursiveCharacterTextSplitter**. This ensures that the embeddings are accurate, as shorter chunks provide better context during the retrieval phase.

### Embedding and Vector Store

The app uses **OpenAIEmbeddings** to convert document chunks into dense vectors, which are stored in **Chroma**, a vector database. Chroma enables **persistent storage** of embeddings, so the vector store doesn't need to be rebuilt every time the app is run.

### Retrieval and QA Chain

Once the vector store is created, the app retrieves relevant document chunks using **similarity-based search**. These documents are passed to the **RetrievalQA chain**, which utilizes **GPT-4** to generate answers. The custom prompt template ensures that the language model stays focused on the retrieved context, further improving the relevance of the answers.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmetybesiroglu/rag-personal-assistant
   cd rag-personal-assistant
   ```

2. Install dependencies using **Poetry**:
   ```bash
   poetry install
   ```

3. Set up your environment variables:
   - Create a `.env` file in the root directory and add your OpenAI API key:
     ```env
     OPENAI_API_KEY=your-openai-api-key
     ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit app in your browser.
2. Select a folder with documents you want the assistant to use.
3. Type a question in the input box, and click **Get Answer**.
4. The app will retrieve relevant documents, generate an answer using GPT-4, and display the conversation history.

### Customization

- **System Prompt**: You can modify the behavior of the assistant by editing the system prompt in the sidebar. This allows you to control how GPT-4 generates answers based on the retrieved documents.
- **Document Refresh**: Use the **Refresh Documents** button to update the vector store when new documents are added.

## Future Improvements

- **Additional Document Formats**: Extend support for more file types like Excel.
- **Cloud Storage Integration**: Load documents from cloud storage platforms (e.g., Google Drive, AWS S3).
- **Advanced Filtering**: Add metadata-based filtering options to refine document retrieval.
- **Interactive Chat Interface**: Enhance the user experience with a more dynamic conversation interface.

