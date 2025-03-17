# CorrelationQA
The official repository of the paper "The Instinctive Bias: Spurious Images lead to Hallucination in MLLMs"

## Installation
```bash
pip install -r requirements.txt
```

## How to use
### 1. Evaluate using your MLLMs
```bash
python eval_llava.py --model_path llava-hf/llava-1.5-7b-hf
```
We implement an toy example for LLaVA model. You can modify the corresponding part in the code for using different MLLMs.

### 2. Calculate the accuracy
```bash
python output.py
```
The average accuracy would be printed in the terminal.