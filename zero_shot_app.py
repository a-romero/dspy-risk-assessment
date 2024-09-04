import dspy
from dspy.evaluate import Evaluate
from dspy import ChainOfThought
from typing import List, Optional
import pandas as pd
from dsp import Claude

# Initialize the language model
#worker = dspy.OpenAI(model="gpt-3.5-turbo", model_type="chat", max_tokens=3000)
worker = Claude(model="claude-3-5-sonnet-20240620", max_tokens=3000)
#worker = dspy.Cohere(model="command-r-plus", max_tokens=3000)
#worker = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.2', max_tokens=3000)
dspy.configure(lm=worker)
dspy.settings.configure(backoff_time = 60)

# Input data
applicant_info = """
Name: John Doe
Age: 35
Annual Income: $75,000
Credit Score: 720
Existing Debts: $20,000 in student loans, $5,000 in credit card debt
Loan Amount Requested: $250,000 for a home mortgage
Employment: Software Engineer at Tech Corp for 5 years
"""


class ZeroShot(dspy.Module):
    """
    You are given a piece of text that contains information about an applicant. 
    Analyze the applicant's financial information and return a risk assessment.
    """
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict("question -> answer")

    def forward(self, applicant):
        return self.prog(question="Analyze the applicant's financial information and return a risk assessment. Applicant: " + applicant)


module = ZeroShot()
response = module(applicant_info)
print(f"ZeroShot response:\n {response}")

prompt_used = worker.inspect_history(n=1)
print(f"Prompt used: {prompt_used}")
