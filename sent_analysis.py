from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith.schemas import Example, Run
from langsmith import Client
import json

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI


class SentimentEvaluator(RunEvaluator):
    def __init__(self):
        prompt = """Is the predominant sentiment in the following statement positive, negative, or neutral?
---------
Statement: {input}
---------
Respond in one word: positive, negative, or neutral.
Sentiment:"""

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.chain = LLMChain.from_string(llm=llm, template=prompt)

    def evaluate_run(self, data: str) -> EvaluationResult:
        response = self.chain({
            "input" : data,
        })
        return response['text']