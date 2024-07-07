'''import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from typing import Any, Dict, Iterable
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_core.embeddings import Embeddings
from document_loader import LoadDocument
from chunker import ChunkData

# Load .env file and API Keys
load_dotenv(find_dotenv(), override=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")
elastic_api_key = os.environ.get("ELASTIC_API_KEY")
elastic_cloud_id = os.environ.get("ELASTIC_CLOUD_ID")
elastic_end_point = os.environ.get("ELASTIC_END_POINT")

data = LoadDocument("../data/ccc.pdf").content
chunks = ChunkData(data, 256, 64).get_splits()
print(chunks[0])

client = Elasticsearch(hosts=elastic_end_point, api_key=elastic_api_key, request_timeout=30, max_retries=10,
                       retry_on_timeout=True)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
index_name = "c"
text_field = "text"
dense_vector_field = "OpenAI-Embeddings"
metadata = 'Metadata'
texts = []
metadatas = []

for i in chunks:
    texts.append(i.page_content)
    metadatas.append(i.metadata)

print(len(texts))
print(len(metadatas))
print(texts)
print()
print(metadatas)


def create_index(
        es_client: Elasticsearch,
        index_name: str,
        text_field: str,
        dense_vector_field: str,
        metadata: str,
):
    es_client.indices.create(
        index=index_name,
        mappings={
            "properties": {
                text_field: {"type": "text"},
                dense_vector_field: {"type": "dense_vector",
                                     "similarity": "cosine"},
                metadata: {
                    "properties": {
                        "source": {"type": "text"},
                        "page": {"type": "integer"},
                        "start_index": {"type": "integer"}
                    }
                },
            }
        },
    )


def index_data(
        es_client: Elasticsearch,
        index_name: str,
        text_field: str,
        dense_vector_field: str,
        embeddings: Embeddings,
        texts: Iterable[str],
        metadatas,
        refresh: bool = True,
) -> None:
    create_index(
        es_client, index_name, text_field, dense_vector_field, metadata
    )

    vectors = embeddings.embed_documents(list(texts))
    requests = []
    print(len(vectors))
    print(vectors[0])
    requests = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": i,
            text_field: text,
            dense_vector_field: vector,
            metadata: metadatas,
        }
        for i, (text, vector, metadatas) in enumerate(zip(texts, vectors, metadatas))
    ]

    bulk(es_client, requests)

    if refresh:
        es_client.indices.refresh(index=index_name)
    return len(requests)


index_data(client, index_name, text_field, dense_vector_field, embeddings, texts, metadatas)
'''