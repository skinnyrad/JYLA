import streamlit as st
import requests
import json
import tempfile
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader  # For Word files

def reset_chat():
    """Reset chat history and document processing"""
    st.session_state.messages = []
    st.session_state.vector_store = None
    if 'qa_chain' in st.session_state:
        del st.session_state.qa_chain

def get_ollama_models():
    """Fetch available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model['name'] for model in response.json()['models']]
            return models
        return []
    except Exception as e:
        st.warning(f"Could not fetch Ollama models: {e}")
        return []

def process_uploaded_file(uploaded_file):
    """Process uploaded documents for several file types: PDF, DOCX, JSON, CSV, TXT, and Markdown."""
    filename = uploaded_file.name.lower()
    if filename.endswith('.pdf'):
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif filename.endswith('.docx'):
        # Save the uploaded DOCX to a temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        # Use Docx2txtLoader to extract text from the Word document [1]
        loader = Docx2txtLoader(tmp_path)
        documents = loader.load()
        text = " ".join(doc.page_content for doc in documents)
        return text
    elif filename.endswith('.json'):
        uploaded_file.seek(0)
        try:
            data = json.load(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing JSON file: {e}")
            return ""
        # Convert the parsed JSON to a formatted string
        text = json.dumps(data, indent=2)
        return text
    elif filename.endswith('.csv'):
        uploaded_file.seek(0)
        # Read CSV file content as text
        text = uploaded_file.getvalue().decode("utf-8")
        return text
    else:  # For .txt and .md files (Markdown)
        uploaded_file.seek(0)
        text = uploaded_file.getvalue().decode("utf-8")
        return text

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "selected_llm" not in st.session_state:
    st.session_state.selected_llm = "llama3.2"
if "selected_embedding" not in st.session_state:
    initial_models = get_ollama_models()
    st.session_state.selected_embedding = (
        "nomic-embed-text" if "nomic-embed-text" in initial_models 
        else initial_models[0] if initial_models else None
    )

# Streamlit UI
st.title("JYLA (Just Your Lazy AI)")
st.subheader("Upload a file")

# Model selection sidebar
with st.sidebar:
    st.header("Model Settings")
    ollama_models = get_ollama_models()
    
    # LLM model selection
    st.session_state.selected_llm = st.selectbox(
        "LLM Model",
        options=ollama_models,
        index=ollama_models.index(st.session_state.selected_llm) 
        if st.session_state.selected_llm in ollama_models else 0
    )
    
    # Embedding model selection with smart default
    if ollama_models:
        embed_index = (
            ollama_models.index("nomic-embed-text:latest") 
            if "nomic-embed-text:latest" in ollama_models
            else 0
        )
        st.session_state.selected_embedding = st.selectbox(
            "Embedding Model",
            options=ollama_models,
            index=embed_index
        )
        
    else:
        st.warning("No embedding models available")

    
    # Reset button
    st.button("Reset Chat & Document", on_click=reset_chat, 
              help="Clear conversation history and unload current document")

# File upload widget now supports multiple files.
uploaded_files = st.file_uploader(
    "Choose document(s)",
    type=["pdf", "txt", "md", "json", "csv", "docx"],
    accept_multiple_files=True
)

# Process files when uploaded (only if a vector store hasn't been created yet)
if uploaded_files and not st.session_state.vector_store:
    with st.spinner("Processing document(s)..."):
        combined_text = ""
        # Iterate over each uploaded file and concatenate its text
        for uploaded_file in uploaded_files:
            combined_text += "\n" + process_uploaded_file(uploaded_file)

        # Use the existing text splitter to split the combined text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_text(combined_text)

        # Generate embeddings and create the vector store
        embeddings = OllamaEmbeddings(model=st.session_state.selected_embedding)
        st.session_state.vector_store = FAISS.from_texts(splits, embedding=embeddings)

        # Create the RAG chain with the selected LLM model and retriever
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            Ollama(model=st.session_state.selected_llm),
            retriever=st.session_state.vector_store.as_retriever(),
            return_source_documents=True
        )
    st.success("Documents processed! Start chatting below")


# Chat Interface
if prompt := st.chat_input("Ask about your document"):
    with st.spinner("Thinking..."):
        # Append the user's new question
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # If there's a previous conversation, append it to the prompt
        if len(st.session_state.messages) > 1:
            previous_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
            instructions = "Answer the question using the provided context.  If the answer isn't provided in the context, try to answer from your own knowledge but don't make up the answer if you don't know."
            prompt = f"{instructions}\n{previous_conversation}\nUser: {prompt}"
        
        # Now process the prompt with the previous conversation included
        if st.session_state.vector_store:
            result = st.session_state.qa_chain.invoke(prompt)
            response = f"{result['result']}"
        else:
            response = "Please upload a document first"
        
        # Append the chatbot's response
        st.session_state.messages.append({"role": "assistant", "content": response})


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
