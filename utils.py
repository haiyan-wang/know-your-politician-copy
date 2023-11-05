import streamlit as st
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.chains import LLMChain

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
    You are an AI chatbot for answering questions about US legislators and politicians.
    You are given the following extracted parts of historical hearings, speeches, and bills. 
    Use what you already know and the additional information given to infer a reasonable opinion they have on the topic provided in the question.  
    For example, you can look at their voting history, bills they've signed, statements they've made to infer their opinion on topics. 
    Don't make up an answer. 
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

def filterDB(query):
	llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
	prompt = '''
You are an AI Chatbot who is a linguistic expert. Given a question related to the political 
opinion of a legislator, you can identify the legislator itself and the topic being addressed.
For instance, a question could be the following:
"What is Senator Ted Cruz's opinion on abortion?" In this sentece, the legislator would be
"Senator Ted Cruz", and the topic being addressed would be "abortion". Your answer should be in the
following format: "{{subject: Ted Cruz, topic: abortion}}".

{context}
Question: {question}
'''
	chain = LLMChain.from_string(llm=llm, template=prompt)
	response = chain({
            "question" : query,
        })
	answer = response['text']
	subject = answer['subject']
	topic = answer['topic']
	embeddings = OpenAIEmbeddings()
	llm = ChatOpenAI(temperature=0)
		
	# Load our local FAISS index as a retriever
	vector_store = FAISS.load_local("faiss_index", embeddings)
	filtered_indices = [i for i, meta in enumerate(vector_store.metadata) 
					 if meta["subject"] == subject and meta["topic"] == topic]
	filtered_vectors = [vector_store[i] for i in filtered_indices]
	return filtered_vectors



