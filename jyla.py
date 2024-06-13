from llama_index.llms.ollama import Ollama
from duckduckgo_search import DDGS
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.web import SimpleWebPageReader
import html2text
import ollama
import re

Settings.llm = llm = Ollama(model="llama3:instruct", request_timeout=120.0, temperature=0.1)
messages = []
messages.append({'role': 'assistant', 'content': "You are a helpful assistant that responds in a formal and concise manner"})
Settings.embed_model = ollama_embedding = OllamaEmbedding(
    model_name="llama3:instruct",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
h = html2text.HTML2Text()
h.ignore_links = True

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
    # Simple heuristic: assume each word is a token.  
    # I recommend using autotokenizer from transfomers for better accuracy
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

prev_response = ""
while True:
    userinput = input("You: ")
    messages.append({'role': 'user', 'content': userinput})
    messages = truncate_messages(messages)
    message_history = convert_messages_to_string(messages)
    prompt = f"Please provide a confidence score from 1-100 about how well you can answer this question\n\n{userinput}\n\n based on the interaction between a large language model and user. Only provide the confidence score and nothing else:\n\n{message_history}"
    response = llm.complete(prompt)
    confidence_match = extract_confidence(str(response))
    confidence = int(confidence_match[0])

    if confidence < 80:
        print("Researching...")
        search = f"Based on this context:\n\n{prev_response}\n\n Please generate a generic search engine query (with no filters) for this question:\n\n{userinput}. Respond with the search engine query and nothing else"
        web_query = str(llm.complete(search))
        results = DDGS().text(web_query.replace("'", "").replace('"', "").replace("\\", ""), max_results=1)

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

        prompt = f"Please answer this question\n\n {userinput}\n\n based on the provided context I found from a website:\n\nContext:\n\n{web_content}"
        content_to_summarize.append({'role': 'user', 'content': prompt})
        summary = ollama.chat(model='llama3:instruct', messages=content_to_summarize, stream=False)

        query = f"Please answer the question:\n\n {userinput}\n\n Based on the following information: {summary['message']['content']}"
        response = ollama.chat(model='llama3:instruct', messages=[{'role': 'user', 'content': query}], stream=False)

        print(response['message']['content'])
        prev_response = str(response['message']['content'])
        messages.append({'role': 'assistant', 'content': prev_response})
        print(f"\n\nReferences: {links}")

    else:
        query = f"Please answer this question:\n\n {userinput}\n\n Based on this interaction between the user and AI assistant: {message_history}"
        response = ollama.chat(model='llama3:instruct', messages=[{'role': 'user', 'content': query}], stream=False)
        print(response['message']['content'])
        prev_response = str(response['message']['content'])
        messages.append({'role': 'assistant', 'content': prev_response})
