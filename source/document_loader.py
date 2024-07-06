# IMPORTS
from langchain_community.document_loaders import PyPDFLoader


class LoadDocument:
    def __init__(self, path):
        self.path = path
        self.state = 0
        try:
            self.content = PyPDFLoader(self.path).load()
        except ValueError:
            print('Error loading the file. Check path!')
        else:
            print('Loading successful!')
            self.state = 1


# Test Code
# data = LoadDocument("../data/ccc.pdf").content
# print(data)
