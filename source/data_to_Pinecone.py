# ************************************************************************
"""Run this only if you want to insert data into Pinecone VectorStore"""
# ************************************************************************

# IMPORTS
from document_loader import LoadDocument
from chunker import ChunkData
from light_model import pc_client


# Load PDF
data = LoadDocument(path="../data/HY-TTC_500_IO_Driver_Manual_V3.4.1.pdf", unstructured=False).load()

# Chunk the data you have loaded.
chunks = ChunkData(data, chunk_size=512, chunk_overlap=128).get_splits()

pc_client.delete_index()
pinecone_index = pc_client.create_index()
pc_client.insert_data(chunks=chunks)
