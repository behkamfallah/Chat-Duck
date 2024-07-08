# IMPORTS
from langchain_community.document_loaders import PyPDFLoader


class LoadDocument:
    def __init__(self, path):
        self.path = path
        self.state = 0
        try:
            self.content = PyPDFLoader(self.path).load()

            # To show how many pages the PDF has.
            print(f"Pdf has {len(self.content)} pages.")
        except ValueError:
            print('Error loading the file. Check path!')
        else:
            print('Loading successful!')
            self.state = 1

    def get_content(self):
        return self.content
