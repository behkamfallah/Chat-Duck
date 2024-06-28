import os
import time
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from typing import Any, Dict, Iterable
from elasticsearch.helpers import bulk
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import format_document
from elasticsearch import Elasticsearch

# Load .env file and API Keys
load_dotenv(find_dotenv(), override=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")
elastic_api_key = os.environ.get("ELASTIC_API_KEY")


# Load the PDF in this section.
def load_document(file):
    from langchain_community.document_loaders import PyPDFLoader
    print(f"Loading {file}")
    data_from_pdf = PyPDFLoader(file).load()
    return data_from_pdf


# data is a list.  Each element is a LangChain Document
# for each page of the PDF with the
# page's content and some metadata
# about where in the document the text came from.
data = load_document("../data/cc.pdf")

# To show how many pages the PDF has.
print(f"Pdf has {len(data)} pages.")


# Split data into chunks.
# Chunk size is by default 256 and Chunk overlap is 64.
def chunk_data(text, chunk_size=256, chunk_overlap=64):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   add_start_index=True, separators=['\n\n', '\n', ' ', ''])
    all_splits = text_splitter.split_documents(text)
    return all_splits

elastic_cloud_id = "2815e5811bbe46c4bab9d804d6754964:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDk1Y2RjNDdiNDUzNDQ2NDlhNDM0NTZkYzI2ZDI3ZjBhJGIwNWMwMDAxZGY3NTQwZTM5NDMwODBiODgxZjY5ODA2"

client = Elasticsearch(
  "https://95cdc47b45344649a43456dc26d27f0a.us-central1.gcp.cloud.es.io:443",
  api_key=elastic_api_key
)

# Chunk data using above function.
chunks = chunk_data(data)

print(chunks[1])
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

resp = client.indices.delete(
    index="workplace_index",
)
print(resp)

documents = ElasticsearchStore.from_documents(
    chunks,
    embeddings,
    index_name="workplace_index",
    es_cloud_id=elastic_cloud_id,
    es_api_key=elastic_api_key,
)


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["source"] = record.get("source")
    metadata["page"] = record.get("page")
    metadata["start_index"] = record.get("start_index")
    return metadata


retriever = documents.as_retriever()

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.1)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible. 
    Each passage has a SOURCE which is the title of the document. When answering, cite source name and page of the passages you are answering from below the answer, on a new line, with a prefix of "SOURCE:" and "Page:" . 

    context: {context}
    Question: "{question}"
    Answer:
    """
)

DOCUMENT_PROMPT = PromptTemplate.from_template(
    """
    ---
    Page: {page}
    SOURCE: {source}
    ---
    """
)


def _combine_documents(
    docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


_context = {
    "context": retriever | _combine_documents,
    "question": RunnablePassthrough(),
}

chain = _context | ANSWER_PROMPT | llm | StrOutputParser()

i = 1
while True:
    print('Write Quit or Exit to stop.')
    q = input(f'Question #{i}: ')
    i = i + 1
    if q.lower() in ['quit', 'exit']:
        print('Exiting...')
        time.sleep(4)
        break

    try:
        ans = chain.invoke(q)
    except:
        continue

    print(f'\nAnswer: {ans}')
    print(f'\n {"-" * 50} \n')