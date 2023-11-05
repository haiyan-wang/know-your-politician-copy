from langsmith.evaluation import EvaluationResult, RunEvaluator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt


class SentimentEvaluator(RunEvaluator):
    def __init__(self):
        prompt = """"
        What is the predominant sentiment in the following opinion?
---------
Statement: {input}
---------
Rate the predominant sentiment from 0 to 1, with 0 being 
        the most negative, 0.5 being neutral, and 1 being the most positive.
Sentiment:"""

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.chain = LLMChain.from_string(llm=llm, template=prompt)
    
    def retrieve_response(self, data: str) -> list[str]:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local("faiss_index", embeddings)
        # TODO: iterate through each year of the vector DB, calling evaluate_run() on each iteration
        # This should then return a list or dict of years mapped to sentiment scores
        # Afterwards, in app.py, we should call a function that creates a 
        # visualization of the sentiment scores over time
        return []

    def evaluate_run(self, data: str) -> EvaluationResult:
        response = self.chain({
            "input" : data,
        })
        return response['text']
    
    def createPlot(self, sent_scores: [int], subject: str, politician: str) -> None:
        x = list(sent_scores.keys())
        y = list(sent_scores.values())
        plt.plot(x, y)
        plt.xlabel("Year")
        plt.ylabel("Favorability towards {}".format(subject))
        plt.title("{}'s Sentiment towards {} over Time".format(politician, subject))
        plt.show()
        return None
