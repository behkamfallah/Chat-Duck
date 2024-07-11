# ************************************************************************
"""Run this only if you want to insert data into Elastic Cloud"""
# ************************************************************************

# IMPORTS
from document_loader import LoadDocument
from unstructured_io_handler import UNSTRUCTURED
from elasticsearchhandler import ELASTICSEARCHHANDLER
from enterprise_model import es_client, index_name, embedding_model, text_field_name, dense_vector_field, metadata_name

# Load PDF, remember to put unstructured as True because later we will pass this document to 'unstructured'.
data = LoadDocument(path="../data/HY-TTC_500_IO_Driver_Manual_V3.4.1.pdf", unstructured=True).load()

# Create client for Unstructured
un_client = UNSTRUCTURED()

# Create requests, these requests will be sent to 'unstructured' server.
requests = un_client.create_requests(data)

# Get results back from 'unstructured' server. Chunking will also be done automatically by 'unstructured'.
elements = un_client.pass_requests_to_unstructured_server(requests)

# Remove header of each page. Pages in pdf may contain header, they are not necessary to keep.
elements = un_client.remove_header(elements)

# Turn those elements to Langchain 'Document' type.
documents = un_client.create_langchain_documents(elements)

texts = []
metadatas = []
for chunk in documents:
    texts.append(chunk.page_content)
    metadatas.append(chunk.metadata)

# Define ES handler
es = ELASTICSEARCHHANDLER(es_client=es_client, index_name=index_name, embedding=embedding_model,
                          text_field=text_field_name, dense_vector_field=dense_vector_field,
                          metadata=metadata_name, texts=texts, metadatas=metadatas)

# Create Index
es.create_index()
