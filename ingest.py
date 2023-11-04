import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_transformers import EmbeddingsRedundantFilter

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load the notion content located in the notion_content folder
reader = DirectoryLoader("./data/joint_hearings", glob="**/*.txt", recursive=True, 
                            show_progress= True, loader_cls=TextLoader)
# reader = CSVLoader("./data/congress.csv")
documents = reader.load()

# Split content into smaller chunks
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n","."],
    chunk_size=1500,
    chunk_overlap=100)
docs = markdown_splitter.split_documents(documents)
print(len(docs))
# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", max_retries=20)
# model_name = "BAAI/bge-small-en"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# hf = HuggingFaceBgeEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )
print('wowowow!')
# Convert all chunks into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save locally to 'faiss_index'
# db = FAISS.from_documents(docs, hf)
db = FAISS.from_documents([docs[0]], embeddings)
print("success!")
for i in range(1, len(docs), 10):
    temp = FAISS.from_documents(docs[i:min(i+10,len(docs))], embeddings)
    time.sleep(1)
    db.merge_from(temp)
    print("success!")
db.save_local("faiss_index")
# print(len(db.get()['documents']))
