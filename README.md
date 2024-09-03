# qRisk

This repository contains a collection of Python scripts that demonstrate different approaches to implementing an AI-powered risk assessment system using the DSPy framework. The project focuses on analyzing financial information of loan applicants and providing risk assessments.

## Table of Contents

1. [Overview](#overview)
2. [Files](#files)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Models](#models)
6. [Training Data](#training-data)
7. [Evaluation](#evaluation)

## Overview

This project showcases four different approaches to implementing a risk assessment AI:

1. Bootstrap Few-Shot Learning
2. Labeled Few-Shot Learning
3. Zero-Shot Learning
4. MIPRO

Each approach uses the DSPy framework and various language models to analyze applicant information and provide risk assessments for loan applications.

## Files

- `zero_shot_app.py`: Implements the Zero-Shot learning approach.
- `labeled_few_shot_app.py`: Implements the Labeled Few-Shot learning approach.
- `bootstrap_few_shot_app.py`: Implements the Bootstrap Few-Shot learning approach.
- `mipro_app.py`: Implements the MIPRO learning approach.

## Installation

To run these scripts, you'll need to install the required dependencies. Create a virtual environment and install the packages using pip:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install dspy pandas dsp-ml
```

## Usage

To run any of the scripts, use the following command:

```bash
python <script_name>.py
```

Replace `<script_name>` with the name of the script you want to run (e.g., `bootstrap_few_shot_app.py`).

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

The Bootstrap Few-Shot approach includes a custom evaluation metric `risk_assessment_metric` that assesses the quality of the generated risk assessments based on correctness and completeness.


---

For more information about the DSPy framework, visit [https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy).


