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
index_name = "squadv2test"
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
llm = ChatOpenAI(model='gpt-4o', temperature=0.1, max_tokens=25)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """Use the following pieces of retrieved context to answer the question.
     Your answer should be very short and precise as possible. Your answer should be at most 2 or 3 words.

    context: 
    {context}

    Question: 
    "{question}"
    
    Answer:
    """
)

#print(hybrid_retriever.invoke('foo')[0].page_content)
def _combine_documents(
        docs, document_separator="\n\n"
):
    doc_strings = [doc.page_content for doc in docs]
    print(doc_strings)
    return document_separator.join(doc_strings)


_context = {
    "context": hybrid_retriever | _combine_documents,
    "question": RunnablePassthrough(),
}


# Define Chain
chain = _context | ANSWER_PROMPT | llm | StrOutputParser()
