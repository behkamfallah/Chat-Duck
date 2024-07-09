import os
from elasticsearchhandler import ELASTICSEARCHHANDLER
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from typing import Any, Dict, Iterable
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import format_document
from elasticsearch import Elasticsearch
from unstructured_client.models import shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
from langchain_core.documents import Document
from unstructured_client import UnstructuredClient


# Load .env file and API Keys
load_dotenv(find_dotenv(), override=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")
elastic_api_key = os.environ.get("ELASTIC_API_KEY")
elastic_cloud_id = os.environ.get("ELASTIC_CLOUD_ID")
elastic_end_point = os.environ.get("ELASTIC_END_POINT")
os.environ["UNSTRUCTURED_API_KEY"] = "Fb9iZkrLrIbRfWqt8nTTbize23FmQD"
unstructured_api_key = os.environ.get("UNSTRUCTURED_API_KEY")

'''# client for Unstructured
un_client = UnstructuredClient(
    api_key_auth=unstructured_api_key,
    # if using paid API, provide your unique API URL:
    server_url="https://api.unstructuredapp.io/general/v0/general",
)

# File Loading
path_to_pdf = "../data/HY-TTC_500_IO_Driver_Manual_V3.4.1.pdf"
with open(path_to_pdf, "rb") as f:
    files = shared.Files(content=f.read(), file_name=path_to_pdf,)

# Creating requests
req = shared.PartitionParameters(
    files=files,
    hi_res_model_name="detectron2_onnx",
    pdf_infer_table_structure=True,
    skip_infer_table_types=[],
    chunking_strategy="by_title",
    max_characters=600,
    overlap=128,
)

# Passing Requests to Unstructured
try:
    resp = un_client.general.partition(req)
    elements = dict_to_elements(resp.elements)
except SDKError as e:
    print(e)

# tables = [el for el in elements if el.category == "Table"]
elements = [el for el in elements if el.category != "Header"]
'''
# Tables edit
'''for i in range(len(tables)):
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        file_obj = StringIO(tables[i].metadata.text_as_html)
        tree = etree.parse(file_obj, parser)
        a = etree.tostring(tree, pretty_print=True).decode()
        tables[i].text = a
    except:
        pass'''

'''documents = []
for element in elements:
    metadata = element.metadata.to_dict()
    documents.append(Document(page_content=element.text, metadata=metadata))'''


# Instantiate Embedding Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# ElasticSearch Client
client = Elasticsearch(hosts=elastic_end_point, api_key=elastic_api_key, request_timeout=30, max_retries=10,
                       retry_on_timeout=True)
# Details of Index
index_name = "my_documents"

# Each document in Index will have these 3 fields:
text_field_name = "Text"
metadata_name = "Metadata"
dense_vector_field = "Embeddings"

# Separate text and metadata
texts = []
metadatas = []
'''for chunk in documents:
    texts.append(chunk.page_content)
    metadatas.append(chunk.metadata)

# Define ES handler
es = ELASTICSEARCHHANDLER(es_client=client, index_name=index_name, embedding=embedding_model,
                          text_field=text_field_name, dense_vector_field=dense_vector_field,
                          metadata=metadata_name, texts=texts, metadatas=metadatas)

# Create Index
es.create_index()
'''


def hybrid_query(search_query: str) -> Dict:
    vector = embedding_model.embed_query(search_query)  # same embeddings as for indexing
    return {
        "query": {
            "match": {
                'Text': search_query,
            },
        },
        "knn": {
            "field": 'Embeddings',
            "query_vector": vector,
        },
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