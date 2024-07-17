import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
import ollama
import re
import tiktoken

# Initialize settings and models
Settings.llm = llm = Ollama(model="llama3:instruct", request_timeout=120.0, temperature=0.1)
Settings.embed_model = ollama_embedding = OllamaEmbedding(
    model_name="llama3:instruct",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Initialize the DuckDuckGo search tool
duckduckgo_tool_spec = DuckDuckGoSearchToolSpec()
duckduckgo_tools = duckduckgo_tool_spec.to_tool_list()
# Create the agent
agent = ReActAgent.from_tools(duckduckgo_tools, llm=llm, verbose=True)

# Prompt Template
prompt_template = """
You are a formal and succinct chatbot with extensive knowledge. Your primary task is to provide accurate and helpful responses to user queries. Follow these guidelines:

1. Always answer directly and confidently, without mentioning sources or context.
2. If relevant information is available in your immediate context, use it to inform your response without explicitly referencing it.
3. If no relevant information is found in the immediate context, draw upon your general knowledge to answer the query.
4. Maintain a consistent tone and level of detail regardless of the information source.
5. If you cannot provide a satisfactory answer to a query, state so clearly and offer to assist with related information if possible.
6. Avoid phrases like 'Based on the provided document' or 'According to the context' in all cases.

Respond to each query as if you inherently possess all necessary information, seamlessly blending any provided context with your general knowledge.
"""


# Initialize the encoding for the model you're using
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': "How can I help you today?"}]
if "prev_response" not in st.session_state:
    st.session_state.prev_response = ""
if "use_internet" not in st.session_state:
    st.session_state.use_internet = False

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
    """
    Count the number of tokens in a single message dictionary.
    """
    role = message['role']
    content = message['content']
    
    # Encode the message content to get the token ids
    token_ids_content = encoding.encode(content)
    token_ids_role = encoding.encode(role)
    
    # Add 4 tokens for the sequence identifiers (e.g. <|endoftext|>)
    num_tokens = len(token_ids_content + token_ids_role) + 4
    
    # Add 2 tokens for the role identifier
    num_tokens += 2
    
    return num_tokens

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

# Sidebar for refresh chat button and internet usage checkbox
with st.sidebar:
    st.session_state.use_internet = st.checkbox("Use Internet", value=st.session_state.use_internet)
    if st.button("Refresh Chat"):
        st.session_state.messages = [{'role': 'assistant', 'content': "How can I help you today?"}]
        st.session_state.prev_response = ""
        st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == 'assistant':
            st.write(message["content"])
        else:
            st.text(message["content"])

# Accept user input
if prompt := st.chat_input("You:"):
    
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    # Display the user's message immediately
    with st.chat_message("user"):
        st.text(prompt)
    
    st.session_state.messages = truncate_messages(st.session_state.messages)
    message_history = convert_messages_to_string(st.session_state.messages)
    #with st.status("Processing...", expanded=False) as status:
    # Generate confidence score
    #    confidence_prompt = f"Please provide a confidence score from 1-100 about how well you can answer this question\n\n{prompt}\n\n based on the interaction between a large language model and user. Only provide the confidence score and nothing else:\n\n{message_history}"
    #    confidence_response = llm.complete(confidence_prompt)
    #    confidence_match = extract_confidence(str(confidence_response))
    #    confidence = int(confidence_match[0])

    if st.session_state.use_internet:
        with st.status("Researching...", expanded=True) as status:
            st.write("Searching for data...")
            search_prompt = f"Please rephrase this question so that it can be process by an AI language model:\n\n{prompt}. Based on this interaction between the user and AI assistant: {message_history}\n\nAnswer with the rephrased question and nothing else."
            rephrased_query = str(llm.complete(search_prompt))
            print(rephrased_query)
            try:
                results = agent.query(rephrased_query)
            except:
                results = "Could not process query.  Please try again..."
            st.write("Analyzing Data...")
            st.session_state.prev_response = str(results)
            st.session_state.messages.append({'role': 'assistant', 'content': st.session_state.prev_response})
            status.update(label="Research complete!", state="complete", expanded=False)

    else:
        with st.status("Analyzing...", expanded=True) as status:
            query = f"{prompt_template}\n\nPlease answer this question:\n\n {prompt}\n\n Based on this interaction between the user and AI assistant: {message_history}"
            response = ollama.chat(model='llama3:instruct', messages=[{'role': 'user', 'content': query}], stream=False)
            st.session_state.prev_response = str(response['message']['content'])
            st.session_state.messages.append({'role': 'assistant', 'content': st.session_state.prev_response})
            status.update(label="Analysis complete!", state="complete", expanded=False)

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.write(st.session_state.prev_response)
