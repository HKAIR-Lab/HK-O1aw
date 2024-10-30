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

We use [Align-Anything](https://github.com/PKU-Alignment/align-anything) framework to conduct SFT training on [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B). The training dataset and hyperparameters used are detailed below.

### Chat Template

Specifically, in SFT training, Q, T, and A in the [training dataset](https://huggingface.co/datasets/HKAIR-Lab/O1aw-sft-16k) are segmented by reserved tokens in the tokenizer of Llama-3.1-8B. For more details, see template:

```
system_prompt: str = ''
user_prompt: str = '<|reserved_special_token_0|>{input}<|reserved_special_token_1|>\n'
assistant_thinking: str = '<|reserved_special_token_2|>{thinking}<|reserved_special_token_3|>\n'
assistant_answer: str = '<|reserved_special_token_4|>{answer}<|reserved_special_token_5|>'
template = system_prompt + user_prompt + assistant_thinking + assistant_answer
```

### Hyperparameters

- model
    ```
    # Pretrained model name or path
    model_name_or_path: meta-llama/Llama-3.1-8B 
    # Whether to trust remote code
    trust_remote_code: True
    # The max token length
    model_max_length: 2048
    ```

- train
    ```
    # The deepspeed configuration
    ds_cfgs: ds_z3_config.json
    # Number of training epochs
    epochs: 3
    # Seed for random number generator
    seed: 42
    # Batch size per device for training
    per_device_train_batch_size: 4
    # Batch size per device for evaluation
    per_device_eval_batch_size: 4
    # The number of gradient accumulation steps
    gradient_accumulation_steps: 16
    # Whether to use gradient checkpointing
    gradient_checkpointing: True
    # Initial learning rate
    learning_rate: 2.e-5
    # Type of learning rate scheduler
    lr_scheduler_type: cosine
    # Ratio of warmup steps for learning rate
    lr_warmup_ratio: 0.03
    # Weight decay coefficient
    weight_decay: 0.0
    # Hyper-parameters for adam optimizer
    adam_betas: [0.9, 0.95]
    # Hyper-parameters for adam epsilon
    adam_epsilon: 1.e-8
    # Enable bfloat 16 precision
    bf16: True
    # Enable float 16 precision
    fp16: False
    # The strategy of evaluation, choosing form [epoch, steps]
    eval_strategy: epoch
    # The evaluation interval in step-wise evaluation case
    eval_interval: 10
    # The max norm of gradient
    max_grad_norm: 1.0
    ```

- dataset
    ```
    # Datasets to use for training
    train_datasets: HKAIR-Lab/O1aw-sft-15k
    # The split of train datasets
    train_split: train
    ```

- output
    ```
    # Pretrained model name or path
    model_name_or_path: null
    # Whether to trust remote code
    trust_remote_code: True
    # The max token length
    model_max_length: 2048
    ```

For other hyperparameters, we use the default value in the [configure file](https://github.com/PKU-Alignment/align-anything/blob/main/align_anything/configs/train/text_to_text/sft.yaml). 

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
