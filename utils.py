import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain():
		"""
        The `load_chain()` function initializes and configures a conversational retrieval chain for
        answering user questions.
        :return: The `load_chain()` function returns a ConversationalRetrievalChain object.
        """
		# Load OpenAI embedding model
		embeddings = OpenAIEmbeddings()
		
		# Load OpenAI chat model
		llm = ChatOpenAI(temperature=0)
		
		# Load our local FAISS index as a retriever
		vector_store = FAISS.load_local("faiss_index", embeddings)
		## look into search_kwargs parameter
		retriever = vector_store.as_retriever(search_kwargs={"k": 5})
		
		# Create memory 'chat_history' 
		memory = ConversationBufferWindowMemory(k=5,memory_key="chat_history")
		
		# Create system prompt
		template = """
    You are an AI assistant for answering questions about US legislators and politicians.
    You are given the following extracted parts of historical hearings, speeches, and bills. 
    If given a question asking about a person's opinion on a topic, please use the extracted parts to infer a reasonable response. 
    Don't make up an answer and cite the document that you derived your response from. 
    If the question is not about US politicians or politics, politely inform them that you are tuned to only answer questions about the US politicians or politics.
    
    {context}
    Question: {question}
    Helpful Answer:"""
		
		# Create the Conversational Chain
		chain = ConversationalRetrievalChain.from_llm(llm=llm, 
				                                          retriever=retriever, 
				                                          memory=memory, 
				                                          get_chat_history=lambda h : h,
				                                          verbose=True)
		
		# Add systemp prompt to chain
		# Can only add it at the end for ConversationalRetrievalChain
		QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)
		chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)
		
		return chain
