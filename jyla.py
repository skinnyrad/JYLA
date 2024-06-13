from llama_index.llms.ollama import Ollama
from duckduckgo_search import DDGS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.web import SimpleWebPageReader
import html2text
import ollama
import re

Settings.llm = llm = Ollama(model="llama3:instruct", request_timeout=120.0, temperature=0.1)
messages = []
Settings.embed_model = ollama_embedding = OllamaEmbedding(
    model_name="llama3:instruct",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
h = html2text.HTML2Text()
h.ignore_links = True
data = SimpleDirectoryReader(input_dir="./initial-data").load_data()
index = VectorStoreIndex.from_documents(data)

def extract_confidence(string):
    return re.findall(r'\d+', string)

def convert_messages_to_string(messages):
    message_history = ""
    for message in messages:
        role = message['role']
        content = message['content']
        message_history += f"{role.capitalize()}: {content}\n"
    return message_history

prev_response=""
while True:
    userinput = input("You: ")
    message_history = convert_messages_to_string(messages)
    prompt = f"Please provide a confidence score from 1-100 about how well you can answer this question\n\n{userinput}\n\n based on the interaction between a large language model and user.  Only provide the confidence score and nothing else:\n\n{message_history}"
    response = llm.complete(prompt)
    confidence_match = extract_confidence(str(response))
    confidence = int(confidence_match[0])
    print(confidence)
    #print(message_history)
    #exit(0)
    if confidence < 65:
        print("Researching...")
        search = f"Based on this context:\n\n{prev_response}\n\n Please generate a generic search engine query (with no filters) for this question:\n\n{userinput}.  Respond with the search engine query and nothing else"
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
        content_to_summarize.append({'role':'user', 'content':prompt})
        summary = ollama.chat(model='llama3:instruct', messages=content_to_summarize,stream=False,)

        query = f"Please answer the question:\n\n {userinput}\n\n Based on the following information: {summary['message']['content']}"
        response = ollama.chat(model='llama3:instruct', messages=[{'role':'user', 'content':query}],stream=False,)

        print(response['message']['content'])
        prev_response = str(response['message']['content'])
        messages.append({'role':'assistant', 'content':prev_response})
        print(f"\n\nReferences: {links}")

    else:
        query = f"Please answer this question:\n\n {userinput}\n\n Based on this this interaction between the user and AI assistant: {message_history}"
        response = ollama.chat(model='llama3:instruct', messages=[{'role':'user', 'content':query}],stream=False,)
        print(response['message']['content'])
        prev_response = str(response['message']['content'])
        messages.append({'role':'user', 'content':userinput})
        messages.append({'role':'assistant', 'content':prev_response})