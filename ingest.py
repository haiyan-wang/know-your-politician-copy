import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load the notion content located in the notion_content folder
reader = DirectoryLoader("./data/senate_hearings", glob="**/*.txt", recursive=True, 
                            show_progress= True, loader_cls=TextLoader)
documents = reader.load()

# Split Notion content into smaller chunks
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n","."],
    chunk_size=1500,
    chunk_overlap=100)
docs = markdown_splitter.split_documents(documents)

# Initialize OpenAI embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Convert all chunks into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save locally to 'faiss_index'
db = FAISS.from_documents(docs, embeddings)
db.save_local("faiss_index")