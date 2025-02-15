# JYLA (Just Your Lazy AI)

**JYLA** is a Streamlit application designed to facilitate interaction with documents through an AI chatbot. It supports various document types including PDF, DOCX, JSON, CSV, Markdown, and plain text, allowing users to ask questions about the document's content.

## Features

- **Document Processing**: Supports multiple file formats for document upload.
- **Chatbot Interface**: Users can ask questions about the document content.
- **Model Selection**: Allows selection of different Ollama models for both language processing and embeddings.
- **Session Management**: Keeps track of chat history and document processing state.

## Setup

### Prerequisites

- Python 3.8+
- Streamlit
- PyMuPDF (fitz)
- langchain libraries
- Ollama server running on `localhost:11434`

### Installation

1. **Install Python Packages**:
   ```
   pip install streamlit PyMuPDF langchain langchain_community requests
   ```

2. **Ensure Ollama Server is Running**:
   - Download and install Ollama from [Ollama's official site](https://ollama.ai/).
   - Start the Ollama server:
     ```
     ollama serve
     ```

3. **Run the Application**:
   ```
   streamlit run your_script_name.py
   ```

## Usage

1. **Upload a Document**: Use the file uploader to select and upload your document.

2. **Select Models**: From the sidebar, choose the LLM and embedding models you wish to use.

3. **Chat with the Document**: 
   - Type your question in the chat input box.
   - JYLA will respond based on the document's content.

4. **Reset Chat**: Use the reset button in the sidebar to clear the chat history and unload the current document.

## Code Structure

- **Imports**: Necessary libraries for Streamlit, document processing, and AI functionalities.
- **Functions**:
  - `reset_chat()`: Clears session state related to chat and document processing.
  - `get_ollama_models()`: Fetches available models from the Ollama server.
  - `process_uploaded_file()`: Handles different file types for document extraction.
- **Streamlit UI**: 
  - Title and file uploader setup.
  - Sidebar for model selection and reset functionality.
  - Chat interface for user interaction.

## Notes

- Ensure that the Ollama server is accessible at `localhost:11434`. If running on a different port or host, adjust the `requests.get` URL in `get_ollama_models()`.
- The application uses session state to manage document processing and chat history, which persists across reruns of the Streamlit app.

## Troubleshooting

- **Model Not Found**: If no models are listed, check if the Ollama server is running and accessible.
- **File Processing Errors**: Ensure the file types are supported and the files are not corrupted.

## License

This project is open-sourced under the MIT license. See the LICENSE file for more details.
