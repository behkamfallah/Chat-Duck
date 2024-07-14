### About the Project
Chat with your PDF

This repository is a 'Chat-with-your-PDF' project using two different implementations, namely Light and Enterprise.


### Prerequisites
Ensure that you have installed the libraries in `requirements.txt` which is located in the `.\source\requirements.txt`.
You can run this code from terminal:
```py
!pip install -r requirements.txt
```


If you get "recursive_guard" error while running the code, try using python 3.11.


If you would like to fork the repository be sure that create an .env file in the ./source and put the API keys in it.
These APIs will be needed if you would like to fully operate this code:
```py
OPENAI_API_KEY='...'
ELASTIC_API_KEY='...'
ELASTIC_CLOUD_ID='...'
ELASTIC_END_POINT='...'
UNSTRUCTURED_API_KEY='...'
UNSTRUCTURED_SERVER_URL='...'
PINECONE_API_KEY='...'
```

### Files and Folders

This repository has three main folders:
1. ```./data``` is the folder you should put your pdf file there.

2. ```./source``` is the folder that consists of ```.py``` files.
This folder has these python files with these usages:
   1. To insert data to databases, use these files:
      
      1. ```data_to_ElasticCloud.py```
      2. ```data_to_Pinecone.py```
      
      Simply specify your file in the line 12 and run the file.
   2. To run the whole application on Streamlit you will need the ```streamlit_app.py```:
         Open Terminal an change directory to ```./source``` and then type:
         ```.py
      streamlit run streamlit_app.py
      ```      
   3. ```document_loader.py``` has the responsibility to Load PDFs. You can call an instance of LoadDocument class that is implemented in this file.
   4. ```chunker.py``` has the responsibility to chunk the data. This file is used only for dealing with the data that will be indexed to Pinecone database.
   5. ```pinecone_handler.py``` handles the client and connection to Pinecone servers. It also retrieves data.
   6. ```elasticsearchhandler.py``` handles the client and connection to Elastic Cloud.
   7. ```unstructured_io_handler.py``` handles the connection and getting results from the 'Unstructured.io' servers.
   8. ```light_model.py``` has the chain related to Light Model.
   9. ```enterprise_model.py``` has the chain related to Enterprise Model.
   10. ```test_synthetic_data.py``` is for testing the app via benchmarks. If you want to run this file, remember to change context window of light model and use ```enterprise_model_for_test.py``` instead of ```enterprise_model.py```.

