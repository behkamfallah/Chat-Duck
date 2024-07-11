# IMPORTS
import os
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared
from unstructured.staging.base import dict_to_elements
from unstructured_client.models.errors import SDKError
from langchain_core.documents import Document


class UNSTRUCTURED:
    # Immediately create a client
    def __init__(self):
        self.unstructured_api_key = os.environ.get("UNSTRUCTURED_API_KEY")
        self.server_url = os.environ.get("UNSTRUCTURED_SERVER_URL")
        self.client = UnstructuredClient(
            api_key_auth=self.unstructured_api_key,
            # if using paid API, provide your unique API URL:
            server_url=self.server_url,
        )
        print('UNSTRUCTURED client created successfully.')

    # Creating requests
    @staticmethod
    def create_requests(files):
        try:
            req = shared.PartitionParameters(
                files=files,
                hi_res_model_name="detectron2_onnx",
                pdf_infer_table_structure=True,
                skip_infer_table_types=[],
                chunking_strategy="by_title",
                max_characters=600,
                overlap=128,
            )
            print('Requests created successfully.')
            return req
        except:
            print('Error in Creating requests.')
            return False

    def pass_requests_to_unstructured_server(self, requests):
        try:
            resp = self.client.general.partition(requests)
            elements = dict_to_elements(resp.elements)
            return elements
        except SDKError as error:
            print(error)
            return False

    @staticmethod
    def remove_header(elements):
        elements = [el for el in elements if el.category != "Header"]
        return elements

    @staticmethod
    def create_langchain_documents(elements):
        documents = []
        for element in elements:
            metadata = element.metadata.to_dict()
            documents.append(Document(page_content=element.text, metadata=metadata))
        return documents


