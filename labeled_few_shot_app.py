# type: ignore

import json
import dspy
from dspy.evaluate import Evaluate
from dspy import ChainOfThought
from dspy.teleprompt import LabeledFewShot
from typing import List, Optional
import pandas as pd
from dsp import Claude

# Initialize the language model
#worker = dspy.OpenAI(model="gpt-3.5-turbo", model_type="chat", max_tokens=3000)
worker = Claude(model="claude-3-5-sonnet-20240620", max_tokens=3000)
#worker = dspy.Cohere(model="command-r-plus", max_tokens=3000)
#worker = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.2')
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

class RiskAssessment(dspy.Signature):
    """Analyze the applicant's financial information and return a risk assessment."""
    question = dspy.InputField()
    applicant = dspy.InputField()
    answer = dspy.OutputField(desc="""
                              A thorough risk analysis about the applicant, justifying the assessment 
                              for each of the parameters considered from the applicant
                              """
                              )

class RiskAssessmentAgent(dspy.Module):
    def __init__(self, role: str, 
                 ):
        self.role = role
        self.question = "Analyze the applicant's financial information and return a risk assessment."
        self.assess_risk = ChainOfThought(RiskAssessment, n=3)
    def forward(self, applicant:str):
        question = self.question
        applicant = applicant
        pred = self.assess_risk(question=question, applicant=applicant)

        return dspy.Prediction(answer = pred.answer)

# Load the training data
dataset = json.load(open("data/training_data.json", "r"))['examples']
trainset = [dspy.Example(question="Analyze the applicant's financial information and return a risk assessment", 
                         applicant=e['applicant'], 
                         answer=e['answer']) for e in dataset]

risk_assessment_agent_role = "Risk Assessment Officer"

# Train
teleprompter = LabeledFewShot()
lfs_optimized_advisor = teleprompter.compile(RiskAssessmentAgent(role=risk_assessment_agent_role), 
                                             trainset=trainset[3:])

response = lfs_optimized_advisor(applicant_info)
print(f"LabeledFewShot Optimised response:\n {response}")

prompt_used = worker.inspect_history(n=1)
print(f"Prompt used: {prompt_used}")
