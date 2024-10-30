# HK-O1-Law

## Training Details

We use [Align-Anything](https://github.com/PKU-Alignment/align-anything) framework to conduct SFT training on [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B). The training dataset and hyper-parameters used are detailed below.

## Dataset Overview

O1aw-Dataset is a comprehensive legal question-answering dataset derived from the   CLIC  , designed to evaluate and enhance legal reasoning capabilities in language models. The dataset follows the O1-style format, featuring complex legal scenarios that require multi-step reasoning.

### Key Features

  Size: 15959 QA pairs with Chain-of-Thought annotations
  Source:    CLIC   （港大法网）
  Language: Simplified Chinese）
  Format: JSON structured data
  Difficulty level: Moderate to Advanced for legal professionals or law students
  Question Categories
  Case Analysis 
  Legal Application
  Legal Concept Explanation

### Reasoning Framework

Each Q-T-A pair includes:
    Detailed question prompt
    3-5 step Chain-of-Thought reasoning, for example:
Validated answer
Quality Assurance
Expert-reviewed reasoning chains
Multi-stage validation process
Reflective verification steps
Consistency checks across similar cases
  Intended Use
  Legal education and training
  Development of legal AI systems
  Assessment of legal reasoning capabilities
  Benchmark for legal language models