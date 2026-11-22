# AI Usage Log

## Project ownership and division of work

**My role:** I led the project and remained responsible for its scope, technical decisions, implementation workflow, execution, evaluation, and interpretation of results. I selected Use Case 4, chose the project plan and model-comparison approach, made the key design decisions, set up the development environment and repositories, ran the experiments on my own laptop and Colab, reviewed the outputs, compared model performance, and tested the final application with my own inputs.

**AI tool used:** Claude (Anthropic), in the Claude desktop app, as a coding assistant and technical support tool. I used it primarily to generate code, help test and debug scripts, explain implementation details, and assist with setup documentation. I reviewed the generated code and outputs, then ran and verified the project work myself.

**Results and verification:** Every numerical result reported in my slides comes from scripts and experiments that I ran myself on my laptop or Colab. Claude did not provide or invent the reported experimental results.

## Work record

| Stage | My work, decisions, and verification | AI assistance |
|---|---|---|
| 0 – Project planning | Chose Use Case 4 and defined the project plan, including the four-model comparison, dataset, and macro-F1 evaluation. Decided to add the edit-distance lexicon matcher, set a 50 ms CPU latency budget, and keep the project in English. Checked the requirements checklist against the assignment brief. | Helped review the assignment brief, draft a requirements checklist (R1–R8, Use Case 4 components, and deliverables), and identify gaps in the initial plan. |
| 1 – Environment and repository | Created the GitHub repository, installed Python 3.11, created the virtual environment, installed packages, handled the Windows Smart App Control issue, ran `env_info.py` locally, and verified the environment and hardware details. | Generated supporting setup files (`requirements.txt`, `.gitignore`, `README`, and `src/env_info.py`), suggested setup steps, and helped diagnose the PyTorch DLL error. |
| 2 – Data preparation | Ran `src/data.py` locally and checked the resulting split sizes against the dataset card. | Generated and tested `src/data.py` and explained its output. |
| 3 – Preprocessing and classical baselines | Ran the lexicon and TF-IDF baseline scripts locally and reviewed the validation results: rule baseline 0.7135 and TF-IDF 0.7007 validation macro-F1. | Generated `src/preprocess.py`, `src/lexicon.py`, and `src/baseline.py`; helped debug-test the scripts and explain the design. |
| 4 – Word2Vec model | Ran `src/w2v_ffn.py` locally and assessed its validation result (0.6412 macro-F1), comparing it with the other baselines. | Generated the Word2Vec feedforward model script, helped debug-test it, and explained the implementation. |
| 5 – DistilBERT fine-tuning | Ran the Colab notebook on a T4 GPU, fine-tuned DistilBERT, uploaded the model to my Hugging Face account, resolved the token-login issue, and reviewed the result (best epoch 1; validation macro-F1 0.7754). | Generated `src/bert_train.py` and the Colab notebook, checked the Transformers API, smoke-tested the script with a small model, and provided setup guidance. |
| 6 – Quantization | Ran `src/quantize.py` locally, downloaded my model from Hugging Face, performed the INT8 quantization experiment, and compared it with FP32 on validation: model size changed from 255.5 MB to 132.3 MB, while validation macro-F1 changed from 0.7754 to 0.7809. | Generated `src/quantize.py`, helped smoke-test it, and explained the quantization method. |
| 7 – Final benchmarking | Ran `src/benchmark.py` locally to evaluate the models on the test set and generate `results/results.csv` (22 Nov, 01:15). | Generated the benchmark script, including the planned metrics and checks, and helped smoke-test it. |
| 7b/8 – Performance tuning and application | Ran the thread sweep and final one-thread benchmark, uploaded the classical models to my Hugging Face repository, launched the Gradio app, and tested it with my own messages. I identified the “stuuupid” example that DistilBERT missed. | Helped diagnose why INT8 was slower than FP32, generated `src/thread_sweep.py`, the shared `CPU_THREADS` setting, `app.py`, and `src/upload_models.py`, and helped test the app with debug models. |
| 9 – README and fresh-environment test | Ran the fresh-environment test: cloned the repository into a new folder, created a new virtual environment with an empty model cache, and confirmed the app downloaded all models and ran. | Updated the README quick start and troubleshooting, and provided the test steps. |
| 10–12 – Diagram, slides, and video | Reviewed the architecture diagram and slides against the code and my results, and recorded and narrated the demo video. | Generated the architecture diagram from the code, drafted the slides from my result files, and drafted the video script. |

## Responsibility summary

I was responsible for the project direction and decisions, environment and repository setup, running all experiments, examining and interpreting the results, comparing model performance, uploading the models, and testing the application. Claude was used as a coding assistant to generate or revise scripts and supporting files, explain technical details, and help identify and fix code or setup issues. The final experimental results were produced by my own runs, not supplied by the AI.
