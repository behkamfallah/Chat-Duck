import os
import time
from document_loader import LoadDocument
from chunker import ChunkData
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from typing import Any, Dict, Iterable
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_elasticsearch import ElasticsearchStore
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import format_document
from elasticsearch import Elasticsearch
from langchain_elasticsearch import DenseVectorStrategy


# Load .env file and API Keys
load_dotenv(find_dotenv(), override=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")
elastic_api_key = os.environ.get("ELASTIC_API_KEY")
elastic_cloud_id = os.environ.get("ELASTIC_CLOUD_ID")
elastic_end_point = os.environ.get("ELASTIC_END_POINT")


# 'data' is a list of LangChain 'Document's.
# Each Document is page of the PDF with the
# page's content and some metadata about where
# in the pdf the text came from.
data = LoadDocument("../data/ccc.pdf").content

# To show how many pages the PDF has.
print(f"Pdf has {len(data)} pages.")

# Chunk data
chunks = ChunkData(data, 256, 64).get_splits()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

client = Elasticsearch(
  hosts=elastic_end_point,
  api_key=elastic_api_key
)
resp = client.indices.delete(index="workplace_index", ignore_unavailable=True)

my_documents = ElasticsearchStore.from_documents(
    chunks,
    embedding=embeddings,
    index_name="workplace_index",
    es_cloud_id=elastic_cloud_id,
    es_api_key=elastic_api_key,
    distance_strategy="COSINE",
    strategy=DenseVectorStrategy(hybrid=True),
    vector_query_field='dense_vector',
    query_field='texts',
)


def hybrid_query(search_query: str) -> Dict:
    vector = embeddings.embed_query(search_query)  # same embeddings as for indexing
    return {
        "query": {
            "match": {
                'texts': search_query,
            },
        },
        "knn": {
            "field": 'dense_vector',
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10,
        },
        "rank": {"rrf": {}},
    }


hybrid_retriever = ElasticsearchRetriever.from_es_params(
    index_name='workplace_index',
    body_func=hybrid_query,
    content_field='texts',
    url=elastic_end_point,
    api_key=elastic_api_key
)


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


def doc_format(list_of_document_objects):
    for document in list_of_document_objects:
        document.metadata['source'] = document.metadata['_source']['metadata']['source']
        document.metadata['page'] = document.metadata['_source']['metadata']['page']
    return list_of_document_objects


def _combine_documents(
    docs, document_prompt=DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


_context = {
    "context": hybrid_retriever | doc_format | _combine_documents,
    "question": RunnablePassthrough(),
}

chain = _context | ANSWER_PROMPT | llm | StrOutputParser()

i = 1
while True:
    print('Write Quit or Exit to stop.')
    user_query = input(f'Question #{i}: ')
    i = i + 1
    if user_query.lower() in ['quit', 'exit']:
        print('Exiting...')
        time.sleep(4)
        break

    ai_answer = chain.invoke(user_query)

    print(f'\nAnswer: {ai_answer}')
    print(f'\n {"-" * 50} \n')