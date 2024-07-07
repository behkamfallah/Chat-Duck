# IMPORTS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Split data into chunks.
# Chunk size is by default 256 and Chunk overlap is 64.
class ChunkData:
    def __init__(self, text, chunk_size, chunk_overlap):
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
                                                            add_start_index=True, separators=['\n\n', '\n', ' ', ''])
        print('Data Chunked.')

    def get_chunk_size(self):
        return self.chunk_size

    def get_chunk_overlap(self):
        return self.chunk_overlap

    def get_splits(self):
        return self.text_splitter.split_documents(self.text)
