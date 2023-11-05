import time
import streamlit as st
from utils import load_chain
from sent_analysis import SentimentEvaluator
import re


evaluator = SentimentEvaluator()
company_logo = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Seal_of_the_United_States_Congress.svg/1201px-Seal_of_the_United_States_Congress.svg.png'

# Configure streamlit page
st.set_page_config(
    page_title="Your Notion Chatbot",
    page_icon=company_logo
)

# Initialize LLM chain in session_state
if 'chain' not in st.session_state:
    st.session_state['chain']= load_chain()
    st.session_state['sentiments'] = []

# Initialize chat history
if 'messages' not in st.session_state:
    # Start with first message from assistant
    st.session_state['messages'] = [{"role": "assistant", 
                                  "content": "Hi! I'm the KnowYourPolitician chatbot. What do you want to learn about today?"}]

# Display chat messages from history on app rerun
# Custom avatar for the assistant, default avatar for user
for message in st.session_state.messages:
    if message["role"] == 'assistant':
        with st.chat_message(message["role"], avatar=company_logo):
            st.markdown(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat logic
if query := st.chat_input("Ask me anything"):
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Regex logic
    sentiment_pattern = r"sentiment|over time|opinion|view|believe|belief|changed|differ|alter|reverse|reversal|shift|switch"
    vote_pattern = r"voting record|voting history|voting patterns|party affiliation|vote in line|vote against|vote on|vote over time"

    run_sentiment = True if re.search(sentiment_pattern, query) else False # True if we want to display sentiment analysis
    run_vote = True if re.search(vote_pattern, query) else False # True if we want to display vote analysis

    with st.chat_message("assistant", avatar=company_logo):
        sentiments = []
        message_placeholder = st.empty()
        # Send user's question to our chain
        result = st.session_state['chain']({"question": query})
        response = result['answer']
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
            if run_sentiment:
                print(True)
                sent = evaluator.evaluate_run(chunk)
                #print(sent)
                sentiments.append(sent)
            else:
                print(False)

        if run_sentiment:
            message_placeholder.markdown(full_response + "\n Here are their opinion on the topic over time:" + sent)
        

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    if run_sentiment:
        st.session_state.sentiments.append({"role": "assistant", "content": sentiments})