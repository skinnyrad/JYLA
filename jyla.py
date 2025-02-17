#!/usr/bin/env python
"""
A Streamlit chat app that processes files and uses LangChain through Ollama.
The conversation history is displayed from top (oldest) to bottom (newest)
and the “Thinking…” spinner appears at the bottom.
"""

import streamlit as st
import requests
import json
import tempfile
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import Docx2txtLoader  # For Word files
import re
from time import sleep

# Add this helper function to parse think tags
def parse_think_tags(response):
    """Extract content from <think> tags and clean final response"""
    thoughts = re.findall(r'<think>(.*?)</think>', response, re.DOTALL)
    clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return thoughts, clean_response

def reset_chat():
    """Reset chat history and document processing"""
    st.session_state.messages = []

def reset_doc():
    st.session_state.vector_store = None
    if "qa_chain" in st.session_state:
        del st.session_state.qa_chain

def get_ollama_models():
    """Fetch available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models
        return []
    except Exception as e:
        st.warning(f"Could not fetch Ollama models: {e}")
        return []

def process_uploaded_file(uploaded_file):
    """Process uploaded documents for PDF, DOCX, JSON, CSV, TXT, and Markdown"""
    filename = uploaded_file.name.lower()
    if filename.endswith('.pdf'):
        text = ""
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif filename.endswith('.docx'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
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
        text = json.dumps(data, indent=2)
        return text
    elif filename.endswith('.csv'):
        uploaded_file.seek(0)
        text = uploaded_file.getvalue().decode("utf-8")
        return text
    else:  # for .txt and .md files
        uploaded_file.seek(0)
        text = uploaded_file.getvalue().decode("utf-8")
        return text

# INITIALIZE SESSION STATE
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
        else initial_models if initial_models else None
    )

# APP TITLE / SIDEBAR / FILE UPLOAD
st.title("JYLA (Just Your Lazy AI)")
st.subheader("Upload a file")

with st.sidebar:
    st.header("Model Settings")
    ollama_models = get_ollama_models()
    st.session_state.selected_llm = st.selectbox(
        "LLM Model",
        options=ollama_models,
        index=ollama_models.index(st.session_state.selected_llm)
            if st.session_state.selected_llm in ollama_models else 0
    )
    if ollama_models:
        embed_index = (ollama_models.index("nomic-embed-text:latest")
                       if "nomic-embed-text:latest" in ollama_models else 0)
        st.session_state.selected_embedding = st.selectbox(
            "Embedding Model",
            options=ollama_models,
            index=embed_index
        )
    else:
        st.warning("No embedding models available")
    st.button("Reset Chat", on_click=reset_chat, 
              help="Clear conversation history")
    st.button("Reset and Embed Documents", on_click=reset_doc, 
              help="Unload current document")

uploaded_files = st.file_uploader(
    "Choose document(s)",
    type=["pdf", "txt", "md", "json", "csv", "docx"],
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.vector_store:
    with st.spinner("Processing document(s)..."):
        combined_text = ""
        for uploaded_file in uploaded_files:
            combined_text += "\n" + process_uploaded_file(uploaded_file)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_text(combined_text)
        embeddings = OllamaEmbeddings(model=st.session_state.selected_embedding)
        st.session_state.vector_store = FAISS.from_texts(splits, embedding=embeddings)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            OllamaLLM(model=st.session_state.selected_llm),
            retriever=st.session_state.vector_store.as_retriever(),
            return_source_documents=True
        )
    st.success("Documents processed! Start chatting below")

# CHAT INTERFACE
# Render the conversation history (oldest at the top)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your document"):
    # FIRST: Immediately display user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # THEN add to session state and process response
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Create assistant response
    with st.chat_message("assistant"):
        if st.session_state.vector_store:
            with st.spinner("Processing your query...", show_time=True):
                # Create new chain with current model each time
                llm = OllamaLLM(model=st.session_state.selected_llm)
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=st.session_state.vector_store.as_retriever(),
                    return_source_documents=True
                )
                result = qa_chain.invoke(prompt)
                response = result["result"]
                thoughts, final_answer = parse_think_tags(response)
                
                #print(f"Thoughts: {thoughts}") # Debugging: print thought tags
                # Display reasoning process if available
                if thoughts:
                    with st.status("Reasoning", expanded=True) as status:
                        # Split thoughts by newline
                        thought_steps = thoughts[0].split('\n')
                        for i, thought in enumerate(thought_steps, 1):
                            if thought.strip():  # Only process non-empty lines
                                st.write(f"**Thought**")
                                st.write(thought.strip())
                                st.write("")  # Add a blank line for readability
                                sleep(1)
                        status.update(label="Reasoning Complete", state="complete", expanded=False)
        
                st.write(final_answer)
            
            # Store only the clean answer in chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer
            })
        else:
            response = "Please upload a document first"
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
    
    #st.rerun()
