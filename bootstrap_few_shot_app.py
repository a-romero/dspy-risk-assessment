# type: ignore

import json
import dspy
from dspy import ChainOfThought
from dspy.teleprompt import BootstrapFewShot
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
    answer = dspy.OutputField(desc="A thorough risk analysis about the applicant, justifying the assessment through each of the parameters considered from the applicant")

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

# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a risk assessment along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

def risk_assessment_metric(gold, pred, trace=None):
    applicant, risk_assessment = gold.applicant, pred.answer

    correct = f"The text above should provide a risk assessment for `{applicant}`. Does it do so? Answer with Yes or No."
    complete = f"Does the text above make for a reasoned assessment across all areas mentioned in `{applicant}`? Answer with Yes or No."

    with dspy.context(lm=worker):
        correct =  dspy.Predict(Assess)(assessed_text=risk_assessment, assessment_question=correct)
        complete = dspy.Predict(Assess)(assessed_text=risk_assessment, assessment_question=complete)

    correct, complete = [m.assessment_answer.split()[0].lower() == 'yes' for m in [correct, complete]]
    score = (correct + complete) if correct else 0

    if trace is not None: return score >= 2
    return score / 2.0

# Load the training data
dataset = json.load(open("data/training_data.json", "r"))['examples']
trainset = [dspy.Example(question="Analyze the applicant's financial information and return a risk assessment", 
                         applicant=e['applicant'], 
                         answer=e['answer']) for e in dataset]

risk_assessment_agent_role = "Risk Assessment Officer"

bfs_trainset = [x.with_inputs('applicant') for x in trainset]
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
bfs_optimized = BootstrapFewShot(metric=risk_assessment_metric, **config)
bfs_optimized_advisor = bfs_optimized.compile(RiskAssessmentAgent(role=risk_assessment_agent_role),
                                              trainset=bfs_trainset)
response = bfs_optimized_advisor(applicant_info)
print(f"BootstrapFewShot Optimised response:\n {response}")

prompt_used = worker.inspect_history(n=1)
print(f"Prompt used: {prompt_used}")

bfs_optimized_advisor.save('bfs_optimized_advisor_compiled.json')
