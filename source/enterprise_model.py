# IMPORTS
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from typing import Dict
from langchain_elasticsearch import ElasticsearchRetriever
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
elastic_cloud_id = os.environ.get("ELASTIC_CLOUD_ID")
elastic_end_point = os.environ.get("ELASTIC_END_POINT")
unstructured_api_key = os.environ.get("UNSTRUCTURED_API_KEY")
unstructured_server_url = os.environ.get("UNSTRUCTURED_SERVER_URL")

# Instantiate Embedding Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# Elastic Client
es_client = Elasticsearch(hosts=elastic_end_point, api_key=elastic_api_key, request_timeout=30, max_retries=10,
                          retry_on_timeout=True)

# Details of Index ------------------------------------------------------------------------------------------
index_name = "my_documents"
# Each document in Index will have these 3 fields, you can check it from Elastic cloud, document tab
text_field_name = "Text"
metadata_name = "Metadata"
dense_vector_field = "Embeddings"
# -----------------------------------------------------------------------------------------------------------


# Hybrid Query, collects retrieval results from both Keyword search and Vector search.
def hybrid_query(search_query: str) -> Dict:
    query_vector = embedding_model.embed_query(search_query)  # Turn user query to Vector.
    return {
        # Keyword Search
        "query": {
            "match": {
                'Text': search_query,
            },
        },
        # Vector Search
        "knn": {
            "field": dense_vector_field,
            "query_vector": query_vector,
        },
        # Re-ranking the results from two retrievers
        "rank": {"rrf": {}},
    }


hybrid_retriever = ElasticsearchRetriever.from_es_params(
    index_name=index_name,
    body_func=hybrid_query,
    content_field=text_field_name,
    url=elastic_end_point,
    api_key=elastic_api_key
)

# Instantiate Language Model
llm = ChatOpenAI(model='gpt-4o', temperature=0.1)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Use the following pieces of retrieved context to answer the question. Answer the question from the retrieved text only. Use exact technical names from retrieved texts and dont change them in your answer. For example
    if a technical name is IO_PWM_SetDuty, use it as it is and don't change it to IO_PWM_SetDutyCycle. Don't makeup names. Be as verbose and educational in your response as possible. 
    Each passage has a SOURCE which is the title of the document. When answering, cite source names and pages of the passages you are using for answering, on a new line, with a prefix of "SOURCE:" and "Page:" . 

    context: {context}

    If and only if the question clearly states a need to a code in C programming language, 
    then make sure you give the code snippet too. Otherwise chat or give answer as you normally do.

    Question: "{question}"
    Answer:
    """
)

DOCUMENT_PROMPT = PromptTemplate.from_template(
    """
    Page: {page}

    Document: {source}
    """
)


def doc_format(list_of_document_objects):
    for document in list_of_document_objects:
        document.metadata['source'] = document.metadata['_source'][metadata_name]['filename']
        document.metadata['page'] = document.metadata['_source'][metadata_name]['page_number']
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

# Define Chain
chain = _context | ANSWER_PROMPT | llm | StrOutputParser()