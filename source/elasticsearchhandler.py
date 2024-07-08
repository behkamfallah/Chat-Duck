# IMPORTS
from elasticsearch.helpers import bulk


class ELASTICSEARCHHANDLER:
    def __init__(self, es_client, index_name: str, embedding, text_field: str,
                 dense_vector_field, metadata: str,
                 texts, metadatas):

        self.es_client = es_client
        self.index_name = index_name
        self.embedding = embedding
        self.text_field = text_field
        self.dense_vector_field = dense_vector_field
        self.metadata = metadata
        self.texts = texts
        self.metadatas = metadatas
        self.requests = []
        self.vectors = []

    def create_index(self):
        try:
            print('Creating Index...')
            self.es_client.indices.create(
                index=self.index_name,
                mappings={
                    "properties": {
                        self.text_field: {"type": "text"},
                        self.dense_vector_field: {"type": "dense_vector",
                                                  "similarity": "cosine"},
                        self.metadata: {
                            "properties": {
                                "source": {"type": "text"},
                                "page": {"type": "integer"},
                                "start_index": {"type": "integer"}
                            }
                        },
                    }
                },
            )
        except:
            print('Could not Create Index!')
        else:
            print('Index created successfully.')
        finally:
            self.index_data()

    def index_data(self, refresh: bool = True) -> None:
        print('Indexing data...')
        self.vectors = self.embedding.embed_documents(list(self.texts))
        self.requests = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": j,
                self.text_field: text,
                self.dense_vector_field: vector,
                self.metadata: metadatas,
            }
            for j, (text, vector, metadatas) in enumerate(zip(self.texts, self.vectors, self.metadatas))
        ]

        try:
            bulk(self.es_client, self.requests)
        except:
            print('Error in Indexing Data. Bulk Error.')
            return
        else:
            print('Data Indexed!')

        if refresh:
            self.es_client.indices.refresh(index=self.index_name)

    def delete_index(self):
        resp = self.es_client.indices.delete(index=self.index_name, ignore_unavailable=True)
        return resp
