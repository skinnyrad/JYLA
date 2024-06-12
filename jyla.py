from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from duckduckgo_search import DDGS
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.web import SimpleWebPageReader
import html2text
import ollama


Settings.llm = llm = Ollama(model="llama3:instruct", request_timeout=120.0, temperature=0.1)

Settings.embed_model = ollama_embedding = OllamaEmbedding(
    model_name="llama3:instruct",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
h = html2text.HTML2Text()
h.ignore_links = True
data = SimpleDirectoryReader(input_dir="./initial-data").load_data()
index = VectorStoreIndex.from_documents(data)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt=(
        "You are a helpful chatbot assistant who responds with succint answers."
    ),
)
prev_response=""
while True:
    userinput = input("You: ")
    search = f"Based on this context:\n\n{prev_response}\n\n Please generate a generic search engine query (with no filters) for this question:\n\n{userinput}.  Respond with the search engine query and nothing else"
    web_query = str(llm.complete(search))
    #print(web_query)
    results = DDGS().text(web_query.replace("'", "").replace('"', "").replace("\\", ""), max_results=1)

    links = []
    content_to_summarize = []
    for result in results:
        if 'href' in result:
            links.append(result['href'])
    web_info = SimpleWebPageReader().load_data(links)
    web_content = h.handle(web_info[0].text)
    prompt = f"Please answer this question\n\n {userinput}\n\n based on the provided context I found from a website:\n\nContext:\n\n{web_content}"
    content_to_summarize.append({'role':'user', 'content':prompt})
    summary = ollama.chat(model='llama3:instruct', messages=content_to_summarize,stream=False,)
    query = f"Please answer the question:\n\n {userinput}\n\n Based on the following context:\nContext: {summary['message']['content']}"
    response = chat_engine.chat(query)
    print(response)
    prev_response = str(response)
    print(f"\n\nReferences: {links}")