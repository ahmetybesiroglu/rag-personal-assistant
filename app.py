import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import hashlib

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# Set OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    st.error("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'")
    st.stop()

# Calculate hash of document directory contents
def get_documents_hash(directory):
    hash_md5 = hashlib.md5()
    for root, _, files in os.walk(directory):
        for file in files:
            with open(os.path.join(root, file), "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

# Load documents from the selected directory
@st.cache_data
def load_documents(directory):
    documents = []
    
    # Load text files
    text_loader = DirectoryLoader(directory, glob='**/*.txt', loader_cls=TextLoader, recursive=True)
    documents.extend(text_loader.load())

    # Load PDF files
    pdf_loader = DirectoryLoader(directory, glob='**/*.pdf', loader_cls=PyPDFLoader, recursive=True)
    documents.extend(pdf_loader.load())

    # Load Word documents
    word_loader = DirectoryLoader(directory, glob='**/*.docx', loader_cls=UnstructuredWordDocumentLoader, recursive=True)
    documents.extend(word_loader.load())

    return documents

# Split documents into chunks
@st.cache_data
def split_documents(_documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(_documents)

# Embed documents and create vector store with persistence
@st.cache_resource
def create_vector_store(_docs, persist_directory):
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = Chroma.from_documents(documents=_docs, embedding=embedding, persist_directory=persist_directory)
    return vector_store

# Define a custom prompt template
def create_chain(retriever, system_prompt):
    llm = ChatOpenAI(model_name='gpt-4', openai_api_key=openai_api_key)

    prompt = PromptTemplate(input_variables=["context", "question"], template=system_prompt)

    # Define the retrieval-based QA chain manually
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain

# Get list of subdirectories
def get_subdirectories(base_dir):
    return ["All"] + [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Main function with chat history
def main():
    st.set_page_config(page_title='Personal Assistant', page_icon='📚')
    st.title('📚 Personal Assistant')

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'system_prompt' not in st.session_state:
        st.session_state['system_prompt'] = (
            "Use the following context to answer the question concisely. "
            "If you don't know the answer, just say you don't know. "
            "Thanks for asking! \n"
            "Context: {context}\n"
            "Question: {question}\n"
            "Helpful Answer:"
        )
    if 'documents_hash' not in st.session_state:
        st.session_state['documents_hash'] = None
    if 'last_result' not in st.session_state:
        st.session_state['last_result'] = None
    if 'selected_folder' not in st.session_state:
        st.session_state['selected_folder'] = 'All'

    # Allow user to modify system prompt
    st.sidebar.header("Customize System Prompt")
    new_system_prompt = st.sidebar.text_area("Edit System Prompt", st.session_state['system_prompt'], height=200)
    if st.sidebar.button("Update System Prompt"):
        st.session_state['system_prompt'] = new_system_prompt
        st.sidebar.success("System prompt updated!")

    # Folder selection
    st.sidebar.header("Select Folder")
    base_dir = 'documents/'
    subdirectories = get_subdirectories(base_dir)
    selected_folder = st.sidebar.selectbox("Choose a folder:", subdirectories, index=subdirectories.index(st.session_state['selected_folder']))

    # Check if selected folder is different from the previously selected one, then refresh documents
    if st.session_state['selected_folder'] != selected_folder:
        st.session_state['selected_folder'] = selected_folder
        st.session_state['documents_hash'] = None  # Clear the documents hash so it reloads
        st.rerun()  # Force the app to rerun and reload documents

    full_path = base_dir if selected_folder == "All" else os.path.join(base_dir, selected_folder)

    if not os.path.exists(full_path):
        st.sidebar.error(f"Folder '{full_path}' does not exist.")
        return

    # Add a button to refresh documents
    if st.sidebar.button("Refresh Documents"):
        st.session_state['documents_hash'] = get_documents_hash(full_path)
        st.rerun()

    # Load and process documents
    with st.spinner('Loading documents...'):
        if st.session_state['documents_hash'] is None:
            st.session_state['documents_hash'] = get_documents_hash(full_path)

        documents = load_documents(full_path)
        if not documents:
            st.warning(f"No documents found in the '{full_path}' folder.")
            return
        docs = split_documents(documents)
        vector_store = create_vector_store(docs, f"db_{st.session_state['documents_hash']}")

    # Initialize the retriever and the chain
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    chain = create_chain(retriever, st.session_state['system_prompt'])

    # User input
    question = st.text_input('Enter your question here:')
    
    if st.button('Get Answer'):
        if question:
            with st.spinner('Searching for the answer...'):
                try:
                    result = chain.invoke({"query": question})
                    answer = result.get("result", "No answer found")
                    
                    # Store question and answer in session state (chat history)
                    st.session_state['history'].append((question, answer))
                    st.session_state['last_result'] = result  # Store the entire result

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning('Please enter a question.')

    # Display the chat history in reverse order
    st.markdown('### Conversation History')
    for i, (q, a) in reversed(list(enumerate(st.session_state['history']))):
        st.write(f"**Q{i+1}:** {q}")
        st.write(f"**A{i+1}:** {a}")

    # Show sources checkbox
    show_sources = st.checkbox('Show Sources')
    if show_sources and st.session_state['last_result']:
        st.markdown('### Source')
        if st.session_state['last_result']['source_documents']:
            source = st.session_state['last_result']['source_documents'][0].metadata.get('source', 'Unknown')
            st.write(f"- {source}")
            st.write("(The answer was derived from different sections of this document)")
        else:
            st.write("No source documents found for this answer.")

if __name__ == "__main__":
    main()
