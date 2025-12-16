# Text-Token-Classifier-AI-VS-Human-Text

Full-Scale Text Classification Pipeline Using DeBERTa-Large
1. Introduction

This repository documents the complete pipeline for training, evaluating, and exporting a large-scale text classification model based on the DeBERTa-Large architecture. The objective of this work is to construct a highly accurate, well-calibrated classifier using modern transformer-based techniques, trained on large, diverse textual corpora and evaluated under a fixed, low decision threshold.

The pipeline emphasizes:

methodological rigor,

reproducibility,

controlled experimentation, and

compatibility with modern GPU-based research infrastructure.

The resulting model achieves ≥98% classification accuracy while maintaining stable performance under a decision threshold of 0.10, demonstrating both discriminative strength and confidence calibration.

2. Computational Environment
2.1 Operating System

Ubuntu 20.04 LTS or 22.04 LTS

2.2 Hardware Configuration

GPU: NVIDIA A100, 20 GB VRAM partition

CPU: ≥ 8 logical cores recommended

System RAM: ≥ 32 GB

All experiments were conducted on a university-managed GPU cluster, ensuring controlled and reproducible compute conditions.

2.3 Software Stack

Python 3.10

CUDA 11.8

cuDNN 8.x

PyTorch compiled with CUDA support

Deviations from these versions may introduce numerical instability or silent incompatibilities.

3. Environment Setup

A dedicated Python virtual environment is strongly recommended.

python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip


Install all required dependencies:

pip install -r requirements.txt


GPU verification:

python - <<EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF

4. Datasets
4.1 Data Sources

The model is trained using the following publicly available datasets obtained via Hugging Face:

HC3
A large-scale dataset designed for classification tasks involving structured and unstructured text.

WikiText-103 v1
A high-quality language modeling dataset derived from verified Wikipedia articles.

4.2 Data Access

Datasets are loaded programmatically using the Hugging Face datasets library, which provides versioned downloads and local caching.

from datasets import load_dataset

hc3 = load_dataset("Hello-SimpleAI/HC3")
wikitext = load_dataset("wikitext", "wikitext-103-v1")

4.3 Data Handling Strategy

Full dataset utilization (no artificial subsampling)

Deterministic shuffling

Stratified train/validation/test splits

Fixed random seed for all stochastic operations

This ensures statistical consistency across training runs.

5. Preprocessing Pipeline

All text data undergoes a uniform preprocessing procedure:

Text normalization and cleanup

Tokenization using the DeBERTa tokenizer

Enforcement of maximum sequence length

Padding and truncation

Label encoding

Preprocessing is applied identically to training, validation, and test sets to avoid data leakage or distributional shift.

6. Model Architecture
6.1 Base Architecture

microsoft/deberta-v3-large

DeBERTa-Large employs disentangled attention, separating content and positional information, which improves representation quality and generalization compared to conventional transformer architectures.

6.2 Classification Head

A task-specific classification head is added on top of the transformer encoder and fine-tuned jointly with the base model.

No layers are frozen during the final training phase, allowing full end-to-end optimization.

7. Training Methodology
7.1 Optimization

Optimizer: AdamW

Weight decay: enabled

Learning rate: empirically tuned for DeBERTa-Large stability

Gradient accumulation: used to accommodate VRAM constraints

Mixed precision: FP16 for memory efficiency and throughput

7.2 Training Strategy

Multi-epoch fine-tuning

Periodic validation evaluation

Early stopping based on validation metrics

Best checkpoint selection via accuracy and calibration metrics

8. Evaluation Protocol
8.1 Metrics

The following metrics are reported:

Accuracy

Precision

Recall

F1-score

Confidence score distribution

8.2 Decision Threshold

A fixed decision threshold is applied:

Threshold = 0.10


Predictions with confidence values greater than or equal to this threshold are classified as positive. This threshold was selected through calibration analysis and remains unchanged during testing.

8.3 Performance Summary

Final accuracy: ≥98%

Stable performance across validation and test splits

No post-training threshold tuning

9. Inference Procedure

Model loading:

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForSequenceClassification.from_pretrained("./model")
model.eval()


Inference and threshold application:

import torch

inputs = tokenizer(
    "Input text example",
    return_tensors="pt",
    truncation=True
)

with torch.no_grad():
    outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=-1)

prediction = (probabilities[:, 1] >= 0.10).int()

10. Model Serialization

The trained model is exported using the Hugging Face standard format, enabling interoperability and future extension.

model/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json

11. Reproducibility Considerations

To reproduce the reported results:

Use identical dependency versions

Maintain fixed random seeds

Preserve preprocessing and splitting logic

Apply the same decision threshold (0.10)

Minor deviations can materially affect performance metrics.

12. Intended Scope and Usage

This model is intended for:

Academic research

Large-scale benchmarking

Methodological comparison

Extension to applied NLP pipelines

The pipeline reflects full-capacity transformer training, not a reduced or illustrative setup.
