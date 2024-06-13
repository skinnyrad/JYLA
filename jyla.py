import streamlit as st
from llama_index.llms.ollama import Ollama
from duckduckgo_search import DDGS
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.web import SimpleWebPageReader
import html2text
import ollama
import re

# Initialize settings and models
Settings.llm = llm = Ollama(model="llama3:instruct", request_timeout=120.0, temperature=0.1)
Settings.embed_model = ollama_embedding = OllamaEmbedding(
    model_name="llama3:instruct",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
h = html2text.HTML2Text()
h.ignore_links = True

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': "How can I help you today?"}]
if "prev_response" not in st.session_state:
    st.session_state.prev_response = ""

# Helper functions
def extract_confidence(string):
    return re.findall(r'\d+', string)

def convert_messages_to_string(messages):
    message_history = ""
    for message in messages:
        role = message['role']
        content = message['content']
        message_history += f"{role.capitalize()}: {content}\n"
    return message_history

def estimate_token_count(text):
    return len(text.split())

def truncate_messages(messages, max_tokens=6000):
    total_tokens = 0
    truncated_messages = []
    for message in messages:
        message_text = f"{message['role'].capitalize()}: {message['content']}\n"
        message_tokens = estimate_token_count(message_text)
        total_tokens += message_tokens
        truncated_messages.append(message)
        if total_tokens > max_tokens:
            break
    while total_tokens > max_tokens and truncated_messages:
        removed_message = truncated_messages.pop(0)
        removed_message_text = f"{removed_message['role'].capitalize()}: {removed_message['content']}\n"
        total_tokens -= estimate_token_count(removed_message_text)
    return truncated_messages

# Streamlit app layout
st.title("JYLA (Just Your Lazy AI)")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("You:"):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    # Display the user's message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages = truncate_messages(st.session_state.messages)
    message_history = convert_messages_to_string(st.session_state.messages)
    
    # Generate confidence score
    confidence_prompt = f"Please provide a confidence score from 1-100 about how well you can answer this question\n\n{prompt}\n\n based on the interaction between a large language model and user. Only provide the confidence score and nothing else:\n\n{message_history}"
    confidence_response = llm.complete(confidence_prompt)
    confidence_match = extract_confidence(str(confidence_response))
    confidence = int(confidence_match[0])

    if confidence < 80:
        with st.status("Researching...", expanded=True) as status:
            st.write("Searching for data...")
            search_prompt = f"Based on this context:\n\n{st.session_state.prev_response}\n\n Please generate a generic search engine query (with no filters) for this question:\n\n{prompt}. Respond with the search engine query and nothing else"
            web_query = str(llm.complete(search_prompt))
            results = DDGS().text(web_query.replace("'", "").replace('"', "").replace("\\", ""), max_results=1)
            st.write("Found URL.")
            links = []
            content_to_summarize = []
            for result in results:
                if 'href' in result:
                    links.append(result['href'])
            try:
                web_info = SimpleWebPageReader().load_data(links)
                web_content = h.handle(web_info[0].text)
            except:
                web_content = ""
            st.write("Analyzing Data...")
            summary_prompt = f"Please answer this question\n\n {prompt}\n\n based on the provided context I found from a website:\n\nContext:\n\n{web_content}"
            content_to_summarize.append({'role': 'user', 'content': summary_prompt})
            summary = ollama.chat(model='llama3:instruct', messages=content_to_summarize, stream=False)
            
            query = f"Please answer the question:\n\n {prompt}\n\n Based on the following information: {summary['message']['content']}"
            response = ollama.chat(model='llama3:instruct', messages=[{'role': 'user', 'content': query}], stream=False)

            st.session_state.prev_response = str(response['message']['content']) + "\n\n" + "References: " + ''.join(links)
            st.session_state.messages.append({'role': 'assistant', 'content': st.session_state.prev_response})
            status.update(label="Research complete!", state="complete", expanded=False)

    else:
        query = f"Please answer this question:\n\n {prompt}\n\n Based on this interaction between the user and AI assistant: {message_history}"
        response = ollama.chat(model='llama3:instruct', messages=[{'role': 'user', 'content': query}], stream=False)
        st.session_state.prev_response = str(response['message']['content'])
        st.session_state.messages.append({'role': 'assistant', 'content': st.session_state.prev_response})

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(st.session_state.prev_response)
