import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


class PineconeHandler:
    def __init__(self, index_name, embedding_model):
        self.index_name = index_name
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.pc_client = Pinecone(api_key=self.pinecone_api_key)
        self.embedding_model = embedding_model

    def delete_index(self):
        try:
            if self.index_name in self.pc_client.list_indexes().names():
                self.pc_client.delete_index(self.index_name)
                print(f'Index:{self.index_name} deleted successfully.')
        except Exception as error:
            print(error)
            pass

    def create_index(self):
        try:
            print('Creating Index...')
            self.pc_client.create_index(
                name=self.index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        except Exception as error:
            print(error)
            print('Error while creating index.')
        else:
            print('Index Created!')
            return self.pc_client.Index(self.index_name)

    def insert_data(self, chunks):
        try:
            print('Inserting data...')
            pinecone_vector_store = PineconeVectorStore.from_documents(chunks, self.embedding_model, index_name=self.index_name)
        except Exception as error:
            print(error)
            print('Error while inserting data.')
        else:
            print('Data inserted successfully!')
            return pinecone_vector_store

    def vector_search(self, query):
        print('Retrieving related chunks...')
        retrieved_docs = self.pc_client.Index(self.index_name).query(
            vector=self.embedding_model.embed_query(query),
            top_k=8,
            include_values=True,
            include_metadata=True
        )
        retrieved_texts = [doc for doc in retrieved_docs['matches'][0]['metadata']['text']]
        return retrieved_texts
