# IMPORTS
import os
from pinecone_handler import PineconeHandler
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load .env file and API Keys
load_dotenv(find_dotenv(), override=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")
elastic_api_key = os.environ.get("ELASTIC_API_KEY")
elastic_cloud_id = os.environ.get("ELASTIC_CLOUD_ID")
elastic_end_point = os.environ.get("ELASTIC_END_POINT")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

# Instantiate Embedding Model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# Name of the index in Elastic Cloud
index_name = "my-documents"

# Pinecone Client
pc_client = PineconeHandler(index_name=index_name, embedding_model=embedding_model)

# Instantiate Language Model
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.1)

# Context Window template
template = """Use the following pieces of retrieved context to answer the question. Answer the question from the retrieved text only. Use exact technical names from retrieved texts and dont change them in your answer. For example
if a technical name is IO_PWM_SetDuty, use it as it is and don't change it to IO_PWM_SetDutyCycle. Don't makeup names. Be as verbose and educational in your response as possible. 

context: {context}

If and only if the question clearly states a need to a code in C programming language, 
then make sure you give the code snippet too. Otherwise chat or give answer as you normally do.

Question: "{q}"
Answer:
"""


prompt = PromptTemplate(
    input_variables=['context', 'q'],
    template=template
)

chain = prompt | llm
