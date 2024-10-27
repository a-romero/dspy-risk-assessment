# DSPy Risk Assessment

This repository contains a collection of Python scripts that demonstrate different approaches to implementing an AI-powered risk assessment system using the DSPy framework. The project focuses on analyzing financial information of loan applicants and providing risk assessments.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Models](#models)
4. [Training Data](#training-data)
5. [Evaluation](#evaluation)

## Overview

This project showcases four different approaches to implementing a risk assessment AI:

1. Zero-Shot
2. Labeled Few-Shot
3. Bootstrap Few-Shot
4. MIPROv2

Each approach uses the DSPy framework and various language models to analyze applicant information and provide risk assessments for loan applications.

## Installation

To run these scripts, you'll need to install the required dependencies. Create a virtual environment and install the packages using pip:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Models

The project is set up to work with various language models:

- Claude 3.5 Sonnet (default)
- OpenAI GPT-3.5 Turbo
- Cohere Command-R Plus
- Hugging Face Mistral-7B-Instruct

To switch between models, uncomment the desired model initialization in the scripts.

## Training Data

The Bootstrap Few-Shot and Labeled Few-Shot approaches use training data stored in `data/training_data.json`. This file should contain examples of applicant information and corresponding risk assessments.

## Evaluation

The Bootstrap Few-Shot and MIPROv2 approaches include a custom evaluation metric `risk_assessment_metric_adv` that assesses the quality of the generated risk assessments based on Bias, Answer Relevancy and Coherence using Deepeval (https://github.com/confident-ai/deepeval). The latter is a custom metric using G-Eval, an advanced evaluation framework for language models that uses LLM-as-a-judge along for tailor-made criteria and evaluation steps and leverages the probabilities of the LLM output tokens to normalize the score through a weighted average.


---

For more information about the DSPy framework, visit [https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy).


