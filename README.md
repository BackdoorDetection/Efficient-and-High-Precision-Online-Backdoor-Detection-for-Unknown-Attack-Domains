# Efficient-and-High-Precision-Online-Backdoor-Detection-for-Unknown-Attack-Domains
This is the code repository for the IWQoS 2026 paper Efficient and High-Precision Online Backdoor Detection for Unknown Attack Domains.

# Backdoor Detection for Large Language Models

This repository provides the implementation of hidden representation extraction
and CNN-based backdoor detection for Transformer-based language models.

The code supports both **encoder models (BERT/DistilBERT)** and
**decoder models (LLaMA)**, and is designed for evaluating and detecting
backdoor attacks in open-domain language models.

The pipeline consists of three stages:

1. Generate poisoned datasets with predefined trigger phrases.
2. Extract intermediate hidden representations from language models.
3. Train CNN-based discriminators for backdoor detection.


## Overview

The overall framework is:

```text
        Text Dataset
             |
             |
    +----------------+
    | Trigger Attack |
    |  Injection     |
    +----------------+
             |
             v
    Poisoned Dataset
             |
             |
 +------------------------+
 | Hidden Representation  |
 | Extraction             |
 +------------------------+
      |              |
      |              |
   BERT           LLaMA
 CLS states    Last-token states
      |              |
      +--------------+
             |
             v
   CNN Backdoor Detector
             |
             v
    Trigger Prediction
```
