# AI Usage Log

**Tool used:** Claude (Anthropic), in the Claude desktop app, as a coding assistant and mentor.
**Rule I follow:** every number in my slides comes from scripts I ran myself on my own laptop or Colab. Claude never supplied results.

| Stage | What Claude did | What I did / decided / verified myself |
|---|---|---|
| 0 | Read the assignment document and drafted REQUIREMENTS.md (checklist of gates, R1–R8, Use Case 4 components, deliverables); pointed out gaps between my plan and the brief. | Chose Use Case 4 and the project plan (4-model ladder, dataset, macro-F1). Decided the open questions: add the edit-distance lexicon matcher, set a 50 ms CPU latency budget, keep the project in English. Checked the checklist against the assignment document. |
| 1 | Wrote requirements.txt (pinned versions, checked compatibility for Windows + Python 3.11), .gitignore, README draft, src/env_info.py; gave setup instructions; diagnosed the PyTorch DLL error. | Created the GitHub repo, installed Python 3.11, created the virtual environment and installed the packages. Turned off Windows Smart App Control myself after it blocked PyTorch. Ran env_info.py on my laptop (result: ALL OK) and confirmed my hardware details. |
| 2 | Wrote src/data.py and tested it; explained the output. | Ran src/data.py on my laptop; checked the split sizes against the dataset card. |
| 3 | Wrote src/preprocess.py (regex normalizer), src/lexicon.py (training-data lexicon + Damerau-Levenshtein matcher) and src/baseline.py (tf-idf + logistic regression, C tuned on validation); debug-tested them on the data; explained the design. | Ran src/lexicon.py and src/baseline.py on my laptop and reviewed the validation results (rule baseline 0.7135, tf-idf 0.7007 val macro-F1). |
| 4 | Wrote src/w2v_ffn.py (own word2vec + feedforward net trained with SGD, tuned on validation) and debug-tested it; explained the design. | Ran src/w2v_ffn.py on my laptop and reviewed the validation results (0.6412 val macro-F1, below both baselines). |
| 5 | Wrote src/bert_train.py and the Colab notebook, checked the transformers 5.24 API and smoke-tested the script with a tiny model; gave Colab instructions. | Ran the Colab notebook on a T4 GPU: fine-tuned DistilBERT and uploaded it to my Hugging Face account; fixed the token login error in the notebook. Best epoch 1, val macro-F1 0.7754. |
| 6 | Wrote src/quantize.py (dynamic INT8 quantization, size and validation-accuracy comparison) and smoke-tested it; explained the method. | Ran src/quantize.py on my laptop: downloaded my model from Hugging Face, quantized it to INT8 and compared it with FP32 on validation (255.5 -> 132.3 MB; val macro-F1 0.7754 -> 0.7809). |
| 7 | Wrote src/benchmark.py (one final test-set evaluation of all models: macro-F1, size, latency over 200 runs, fingerprint check, failure examples) and smoke-tested it. | _TODO (you)_ |
