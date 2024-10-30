# HK-O1-Law

[包括模型介绍，参数，预期的场景@Sirui Han，讲一讲背景]
6 ~ 7 行

## Examples

放 3 - 5 典型的case

Dataset Overview

O1aw-Dataset is a comprehensive legal question-answering dataset derived from the CLIC, designed to evaluate and enhance legal reasoning capabilities in language models. The dataset follows the O1-style format, featuring complex legal scenarios that require multi-step reasoning.

## Our training dataset

Size: 15959 QA pairs with Chain-of-Thought annotations
  Source:    CLIC   （港大法网）
  Language: Simplified Chinese）
  Format: JSON structured data
  Difficulty level: Moderate to Advanced for legal professionals or law students
  Question Categories
  Case Analysis 
  Legal Application
  Legal Concept Explanation

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


## Training details

We use [Align-Anything](https://github.com/PKU-Alignment/align-anything) framework to conduct SFT training on [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B). The training dataset and hyper-parameters used are detailed below.

<SFT>

## Evaluation for HK Laws

In the planning, coming soon.


## Our teams


## Citation
Please cite the repo if you use the data or code in this repo.

```bibtex
@misc{align_anything,
  author = {HKAIR Lab},
  title = {HK-O1-Law Models: A HK Law Large Language Model using O1's Slow Thinking},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HKAIR-Lab/HK-O1-Law}},
}
```


## License

HK-O1-Law models and datasets are released under Apache License 2.0.
