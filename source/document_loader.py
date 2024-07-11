# IMPORTS
from langchain_community.document_loaders import PyPDFLoader
from unstructured_client.models import shared


class LoadDocument:
    def __init__(self, path: str, unstructured=False):
        self.path = path
        self.unstructured = unstructured
        self.content = ''

    def load(self):
        try:
            if self.unstructured:
                with open(self.path, "rb") as f:
                    self.content = shared.Files(content=f.read(), file_name=self.path)
            else:
                self.content = PyPDFLoader(self.path).load()
                # To show how many pages the PDF has.
                print(f"Pdf has {len(self.content)} pages.")
        except ValueError:
            print('Error loading the file. Check path!')
        else:
            print('Loading file was successful!')
            return self.content
