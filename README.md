# JYLA (Just Your Lazy AI)

JYLA is a local Chatbot application that allows you to interact with a large language model while keeping your data private. It also has the ability to check the web for questions that require up-to-date answers.

## Features

- **Private Interaction**: Communicate with a large language model without compromising your data privacy.
- **Modern RAG and AI Technologies**: JYLA was built using llamaindex, Ollama, and llama3.
- **Web Integration**: Optionally search the web for the most current information to answer your questions.
- **Streamlit Interface**: User-friendly interface built with Streamlit.

## Installation

To run JYLA locally, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/skinnyrad/JYLA
    cd JYLA
    ```

2. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Install Ollama and download llama3:instruct**:

    [Install Ollama](https://ollama.com/) and download your desired large language model.  The model used by JYLA is [llama3:instruct](https://ollama.com/library/llama3:instruct).  This model can be download from the command line:

    ```sh
    ollama pull llama3:instruct
    ``` 

4. **Run the Streamlit app**:
    ```sh
    streamlit run jyla.py
    ```

![JYLA-Main](./img/jyla-main.png)

## Usage

1. **Start the Application**: Open your browser and navigate to `http://localhost:8501` to access the JYLA interface.
2. **Interact with the AI**: Type your questions in the chat input and receive responses from the AI.
3. **Use Internet Option**: Click the "Use Internet" checkbox in the sidebar to allow the AI to search the web for up-to-date information if it is not confident in its pretrained knowledge.
4. **Refresh Chat**: Use the "Refresh Chat" button in the sidebar to reset the conversation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Llamaindex](https://www.llamaindex.ai/)
- [Ollama](https://ollama.com/)
- [DuckDuckGo Search](https://duckduckgo.com/)
- [html2text](https://github.com/Alir3z4/html2text)