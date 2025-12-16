# Full-Scale Text Classification Pipeline Using DeBERTa-Large

## 1. Introduction
This repository provides a complete, reproducible pipeline for training, evaluating, and exporting a large-scale text classification model based on the DeBERTa-Large architecture. The model is trained on large benchmark datasets and optimized to achieve high classification accuracy (≥98%) under a fixed decision threshold of 0.10.

The work emphasizes academic rigor, controlled experimentation, and reproducibility, making it suitable for university research, thesis work, and large-scale benchmarking.

## 2. Computational Environment

### Operating System
- Ubuntu 20.04 LTS or 22.04 LTS

### Hardware
- GPU: NVIDIA A100 (20 GB VRAM partition)
- CPU: ≥ 8 cores recommended
- RAM: ≥ 32 GB

### Software Stack
- Python 3.10
- CUDA 11.8
- cuDNN 8.x

## 3. Environment Setup

Create a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify GPU availability:

```bash
python - <<EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF
```

## 4. Datasets

### Data Sources
- HC3 (Hugging Face)
- WikiText-103 v1 (Hugging Face)

### Dataset Loading

```python
from datasets import load_dataset

hc3 = load_dataset("Hello-SimpleAI/HC3")
wikitext = load_dataset("wikitext", "wikitext-103-v1")
```

### Data Handling
- Full dataset usage
- Deterministic shuffling
- Stratified splits
- Fixed random seed

## 5. Preprocessing Pipeline
1. Text normalization
2. Tokenization using DeBERTa tokenizer
3. Maximum sequence length enforcement
4. Padding and truncation
5. Label encoding

The preprocessing pipeline is identical for training, validation, and testing.

## 6. Model Architecture
- Base model: microsoft/deberta-v3-large
- End-to-end fine-tuning with a task-specific classification head
- No layer freezing in the final training phase

## 7. Training Methodology
- Optimizer: AdamW
- Mixed precision (FP16)
- Gradient accumulation for VRAM efficiency
- Validation-based early stopping
- Best checkpoint selection

## 8. Evaluation Protocol
Metrics:
- Accuracy
- Precision
- Recall
- F1-score

Decision threshold:
```
0.10
```

Final performance:
- Accuracy ≥ 98%

## 9. Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")
model.eval()

inputs = tokenizer("Example input text", return_tensors="pt", truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)

prediction = (probs[:, 1] >= 0.10).int()
```

## 10. Model Export

The trained model is saved in Hugging Face format:

```
model/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
```

## 11. Reproducibility Notes
To reproduce results:
- Use identical dependency versions
- Preserve random seeds
- Maintain the same preprocessing and threshold

## 12. Intended Use
- Academic research
- Benchmarking
- Large-scale experimentation
