# SWE015 Homework: Real-Time Chat Moderation (Use Case 4)

**Student:** Mohamed Ahmed ELSAYED · 220911956
**Course:** SWE015 – Introduction to Large Language Models

> Work in progress. The full setup steps and the single run command are added in Stage 9
> and tested in a fresh virtual environment.

## What this is
A chat-style app that labels each message **clean / offensive** live, masks flagged
messages, and logs response time per message. It compares four models on the same
held-out test set:

1. Classical baseline: regex normalization + tf-idf + logistic regression
   (plus an edit-distance lexicon matcher as a rule baseline)
2. Own neural network: word2vec trained on our training messages + a feedforward network (SGD)
3. Transformer: fine-tuned `distilbert-base-uncased`
4. Quantized Transformer: the same DistilBERT with dynamic INT8 quantization (the app's default)

Latency budget: **50 ms per message on CPU** for the quantized model.

## Quick environment check (Stage 1)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.env_info
```

## Data
TweetEval, `offensive` subset (English, from OffensEval 2019 / OLID). Full citations follow in Stage 2.
