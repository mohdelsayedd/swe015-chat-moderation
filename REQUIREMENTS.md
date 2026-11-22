# REQUIREMENTS.md — SWE015 Homework: Building an Intelligent Application

Source: `LLM_HOMEWORK_22_NOV.docx` (read in full, 21 Nov 2026).
Use case chosen: **Use Case 4 — Real-Time Chat Moderation** (English).
Due: **Sunday 22 November 2026**. Weight: 15%. Grades: 0 / 50 / 100.

**How to read this file**
- IDs `R1`–`R8` are the document's own IDs. Every other ID (`UC4-*`, `T-*`, `P-*`, `V-*`, `REPO-*`, `RULE-*`, `G1-*`, `G2-*`) is ours, added only so we can tick items off. The doc section each one comes from is given in brackets.
- Text in "quotes" is the document's exact wording.
- **Gate** column: `0` = failing it gives grade 0 (Gate 1); `50` = failing it caps the grade at 50 (Gate 2); `—` = required by the document but not named in a gate.
- **Stage** = the stage of our plan where the item gets done.
- Status: ☐ not done · ◐ in progress · ☑ done (verified)

---

## 1. Grading gates (§8)

### Gate 1: any of these → grade 0
| ID | Exact wording | Status |
|---|---|---|
| G1-1 | "A deliverable (presentation, video, repository) is missing, or a link cannot be opened." | ☐ |
| G1-2 | "The video does not show the application running with live input (only slides, screenshots or notebook output)." | ☐ |
| G1-3 | "There is no Transformer model, or no quantized model at all." | ☐ |
| G1-4 | "The homework does not address one of the five use cases." | ☐ |
| G1-5 | "Fabricated results, plagiarism, or work copied from another student or an online repository." | ☐ |

### Gate 2: passed Gate 1, but any of these → grade 50
| ID | Exact wording | Status |
|---|---|---|
| G2-1 | "One of the four technique areas is missing, or there is no technique map slide." | ☐ |
| G2-2 | "A model of the ladder is missing, or the models are not evaluated on the same held-out test set." | ☐ |
| G2-3 | "The results table is incomplete (task metric, size or latency missing)." | ☐ |
| G2-4 | "The architecture diagram is missing or does not match the code." | ☐ |
| G2-5 | "The presentation does not follow the Introduction / Development / Conclusion structure." | ☐ |
| G2-6 | "The application does not run the quantized model." | ☐ |
| G2-7 | "The repository cannot be set up from the README." | ☐ |
| G2-8 | "The AI usage slide is missing." | ☐ |
| G2-9 | "The video has no narration, or does not show the architecture and results." | ☐ |

"All requirements met → 100." Also: "A polished interface or very high accuracy does not earn extra points … A Transformer that does not beat the baseline is a valid finding if you explain why."

---

## 2. Core requirements (§5)

| ID | Exact wording | Gate | Stage | Status |
|---|---|---|---|---|
| R1 | "Working application. An interactive interface (Gradio, Streamlit, or a web front end over FastAPI) that accepts new input typed, drawn or uploaded during the demo and returns a result. A notebook that prints outputs is not an application." | 0 (via G1-2) | 8 | ☑ |
| R2 | "Required techniques. At least one technique from each of the four areas in Section 3, shown on a technique map slide." | 50 | 3–6, 11 | ☑ |
| R3 | "Model ladder. Four models evaluated on the same held-out test set: classical baseline, a neural network you trained, a Transformer, and the quantized Transformer." | 50 (0 if no Transformer/quantized) | 3–7 | ☑ |
| R4 | "Results table. One row per model; columns: task metric, model size (MB), latency (ms, averaged over ≥100 runs; state the hardware)." (The CNN before/after rows apply only to Use Cases 1 and 3, so not to us.) | 50 | 7 | ☑ |
| R5 | "Quantized model in the application. The application runs the quantized Transformer. Showing the other models side by side is encouraged." | 50 | 8 | ☑ |
| R6 | "Architecture diagram. Every component and the data flow from user input to output: preprocessing, each model (name and precision), post-processing, user interface, and where it runs. It must match the code." | 50 | 10 | ☑ |
| R7 | "Code repository. GitHub or GitLab, public or shared with the instructor. A README with setup steps and a single run command, a requirements.txt, and no large weight files in the repository (link to Hugging Face Hub or Google Drive instead)." | 50 (0 if missing) | 1, 9 | ☑ |
| R8 | "AI usage statement. One slide, as described in Section 6." | 50 | 11 | ☐ |

---

## 3. Use Case 4 specifics (§4, "Use Case 4 — Real-Time Chat Moderation")

Scenario: "A gaming platform must flag offensive messages in Turkish (or English) within a strict time limit on a CPU-only server."

### 3a. What the application must do
| ID | Exact wording (split into checkable parts) | Stage | Status |
|---|---|---|---|
| UC4-APP-1 | "provide a chat-like interface" | 8 | ☑ |
| UC4-APP-2 | "where each message is labeled live (clean / offensive) with confidence" | 8 | ☑ |
| UC4-APP-3 | "flagged messages are masked" | 8 | ☑ |
| UC4-APP-4 | "a side panel logs the response time of every message" | 8 | ☑ |
| UC4-APP-5 | "and lets the user switch between models" | 8 | ☑ |

### 3b. Required components
| ID | Area | Exact wording | Our component | Stage | Status |
|---|---|---|---|---|---|
| UC4-B1 | B | "regex normalization of noisy text (leetspeak, repeated letters, Turkish characters)" | `src/preprocess.py`: leetspeak, repeated letters, and Turkish-character folding (kept cheaply, though the project is English) | 3 | ☑ |
| UC4-B2 | B | "an edit-distance lexicon matcher as a rule baseline" | **Decided (Q1 = yes):** lexicon built from the **training split only** (words most associated with the offensive label), with fuzzy matching by Levenshtein distance; reported as an extra row "Rule baseline" | 3 | ☑ |
| UC4-B3 | B | "a tf-idf baseline" | `src/baseline.py`: tf-idf + logistic regression | 3 | ☑ |
| UC4-A1 | A | "your own word2vec trained on the training messages, feeding a feedforward network or a 1D CNN" | `src/w2v_ffn.py`: gensim word2vec + PyTorch FFN | 4 | ☑ |
| UC4-C1 | C | "BERTurk or a multilingual BERT fine-tuned for classification" | `distilbert-base-uncased`. This is a substitution, so it needs a justification (see notes) | 5 | ☑ |
| UC4-D1 | D | "a latency budget you set (e.g., 50 ms per message on CPU), met with quantization (optionally also distillation or pruning)" | **Decided (Q2):** budget = **50 ms per message, CPU, quantized DistilBERT** (fixed before any measurement). Reported honestly whether met or not | 6–7 | ☑ |
| UC4-M1 | — | "Metrics: macro-F1 (classes are imbalanced), latency per message." | `src/benchmark.py` → `results/results.csv` | 7 | ☑ |
| UC4-DATA | — | "Suggested data: OffensEval-TR 2020 (Turkish) or Jigsaw Toxic Comment Classification (English)." | `tweet_eval/offensive` (the English OffensEval 2019 / OLID data). This is allowed: "You may change domain details (language, product category, corpus)" | 2 | ☐ |

Substitution rule (§3): "You may replace a component with another technique from the same area if you explain why."
Ladder rule (§4): "You may change domain details … but the application requirements stay the same."

### 3c. Technique areas (§3), with at least one per area
| ID | Area | Our technique(s) | Status |
|---|---|---|---|
| T-A | A · Neural networks & deep learning | "feedforward network trained with SGD and backpropagation" → word2vec + FFN | ☑ |
| T-B | B · Classical NLP | regex normalization; tf-idf; word2vec embeddings; (edit distance, if Q1 = yes) | ☑ |
| T-C | C · Transformers | "fine-tuning BERT … with a task head"; WordPiece subword tokenizer | ☑ |
| T-D | D · Efficient models | "quantization (mandatory)"; "benchmarking size, latency and accuracy" | ☑ |
| T-MAP | — | "Your presentation must include a technique map slide that shows which component of your application uses which technique." | ☑ |

---

## 4. Rules (§6)

| ID | Exact wording | Gate | Status |
|---|---|---|---|
| RULE-1 | "This is an individual homework. … code, results and slides must be your own." | 0 (G1-5) | ☐ |
| RULE-2 | "AI coding assistants are allowed … Add one AI Usage slide stating which tools you used and for which parts." → kept accurate in `AI_USAGE_LOG.md` | 50 (G2-8) | ◐ |
| RULE-3 | "You are responsible for every line in your repository and must be able to explain any component in your architecture diagram if asked." | — | ☐ |
| RULE-4 | "Every number in your presentation must come from your own runs. Numbers taken from an AI answer, a paper or a model card instead of being measured count as fabricated results." | 0 (G1-5) | ☐ |
| RULE-5 | "Use public datasets or data you collect yourself, and cite every source." | — (counts toward plagiarism) | ☐ |

---

## 5. Deliverables (§7)

### 5a. Presentation: P (§7.1) "PDF, 8–12 slides"
| ID | Exact wording | Gate | Stage | Status |
|---|---|---|---|---|
| P-1 | Format: "PDF, 8–12 slides" | 0 if missing (G1-1) | 11 | ☐ |
| P-2 | "The first slide shows your name, student ID, chosen use case, repository link and video link." | — (links: G1-1) | 11 | ☐ |
| P-3 | "Introduction — the problem and who it is for, the chosen use case, and what success looks like." | 50 (G2-5) | 11 | ☐ |
| P-4a | Development: "data (source, size, splits, preprocessing)" | 50 (G2-5) | 11 | ☐ |
| P-4b | Development: "the architecture diagram (R6)" | 50 | 10–11 | ☐ |
| P-4c | Development: "the technique map (R2)" | 50 | 11 | ☐ |
| P-4d | Development: "the four models and how each was trained" | 50 | 11 | ☐ |
| P-4e | Development: "the quantization method" | 50 | 11 | ☐ |
| P-4f | Development: "the results table (R4)" | 50 | 11 | ☐ |
| P-5a | Conclusion: "what the comparison showed (where each model wins and loses)" | 50 (G2-5) | 11 | ☐ |
| P-5b | Conclusion: "limitations with at least one failure example" | 50 | 11 | ☐ |
| P-5c | Conclusion: "what you would do next" | 50 | 11 | ☐ |
| P-5d | Conclusion: "the AI usage slide (R8)" | 50 (G2-8) | 11 | ☐ |

### 5b. Video: V (§7.2) "Two-Minute Demo Video (Link)"
| ID | Exact wording | Gate | Stage | Status |
|---|---|---|---|---|
| V-1 | "A screen recording with voice narration." | 50 (G2-9) | 12 | ☐ |
| V-2 | "Only the first 2:00 are watched." | (content after 2:00 doesn't count) | 12 | ☐ |
| V-3 | "Upload it to YouTube (unlisted) or Google Drive ("anyone with the link") and check the link in a private browser window." | 0 (G1-1) | 12–13 | ☐ |
| V-4 | "The video must show the application running live with new input" | 0 (G1-2) | 12 | ☐ |
| V-5 | "live demo: at least two different inputs, including one difficult case" (0:15–1:15) | — (suggested script) | 12 | ☐ |
| V-6 | "architecture diagram and results table" (1:15–1:45) | 50 (G2-9) | 12 | ☐ |
| V-7 | Suggested timing: 0:00–0:15 problem · 0:15–1:15 demo · 1:15–1:45 architecture + results · 1:45–2:00 conclusion | — | 12 | ☐ |

### 5c. Repository: REPO (§7.3)
| ID | Exact wording | Gate | Stage | Status |
|---|---|---|---|---|
| REPO-1 | "Put the link on the first slide and in the submission form." | 0 if missing/unopenable (G1-1) | 11, 13 | ☐ |
| REPO-2 | README with setup steps + single run command; requirements.txt; no large weights (= R7) | 50 (G2-7) | 9 | ☐ |

---

## 6. Pre-submission checklist (§9, verbatim)
- ☐ The application accepts new input and returns a result (R1).
- ☐ All four technique areas are used, and the technique map slide shows where (R2).
- ☐ Baseline, neural network, Transformer and quantized Transformer are evaluated on the same test set (R3).
- ☐ The results table has task metric, size and latency for every model (R4).
- ☐ The application runs the quantized Transformer (R5).
- ☐ The architecture diagram matches the code (R6).
- ☐ The README setup and run command work in a fresh environment (R7).
- ☐ The AI usage slide is included (R8).
- ☐ The slides follow Introduction / Development / Conclusion, and the first slide has both links.
- ☐ The video is at most 2:00, narrated, and shows the live application, architecture and results.
- ☐ The video and repository links open in a private browser window.

---

## 7. Gaps between the original plan and the document

| # | Gap | Severity | Proposed fix |
|---|---|---|---|
| GAP-1 | Plan has no **edit-distance lexicon matcher** (UC4-B2) | Low | ☑ Resolved: added (Q1) |
| GAP-2 | Plan has no **latency budget** (UC4-D1) | Medium | ☑ Resolved: 50 ms/message on CPU (Q2) |
| GAP-3 | App plan is missing **confidence**, **masking** of flagged messages, and a **side panel logging every message's response time** (UC4-APP-2/3/4) | Medium. "the application requirements stay the same" | Built into Stage 8 |
| GAP-4 | DistilBERT replaces "BERTurk or a multilingual BERT" | Low, if justified | Justification on the technique map slide: English data → English encoder; DistilBERT is BERT distilled to 6 layers, which suits a CPU latency budget |
| GAP-5 | Area A wording is "trained with **SGD** and backpropagation" | Low | Use `torch.optim.SGD` (with momentum) for the FFN, not Adam |
| GAP-6 | Data source must be **cited** (RULE-5) | Low | Cite TweetEval (Barbieri et al., 2020) and OffensEval 2019 / OLID (Zampieri et al., 2019) in the README and on the data slide |
| GAP-7 | R7 forbids weights in the repo, but the README must run from scratch | Medium | App downloads weights from a Hugging Face Hub model repo on first run. Details in Stages 6 and 9 |
| GAP-8 | Your own prompt's summary of Gate 2 left out some items (technique map, Intro/Dev/Conclusion structure, app not running quantized model, video missing narration/architecture/results) | Info | All are now tracked above (G2-1 … G2-9) |

---

## 8. Decisions log
| # | Decision | Why (viva line) |
|---|---|---|
| D1 | Language: English, final | Switching later means redoing the data, all models and the benchmark |
| D2 | Lexicon matcher built from training data only | No outside word list to cite, and no test-set leakage |
| D3 | Latency budget 50 ms/message (CPU, quantized model) | The brief's own example; fixed before measuring so it can't be tuned to the results |
| D4 | Exact version pins; torch 2.15.0 | `quantize_dynamic` is deprecated upstream (moving to torchao) but present in 2.15.0, and pinning keeps it there |
| D5 | FFN optimizer = SGD (+ momentum) | Area A literally says "trained with SGD and backpropagation" |
| D6 | Benchmark machine: Intel Core i7-8550U @ 1.80 GHz, 8 logical cores (torch uses 4 threads), 15.8 GB RAM, Windows (confirm the edition in Settings → About), Python 3.11.9, torch 2.15.0+cpu. Source: `python -m src.env_info` output, 21 Nov | R4 says "state the hardware"; taken from a script, not from memory |
| D7 | Quantized engine on this CPU: `onednn` (the only one listed); `quantize_dynamic` test passed | Stage 6 runs on this backend |
| D8 | Windows Smart App Control blocked torch DLLs (WinError 4551); Mohamed turned it off on his own laptop | Mention in README troubleshooting (Stage 9) |
| D9 | Test set = official TweetEval test split (860 tweets, 27.9% offensive), fingerprint `ac3947edadffd9ab`; 4 exact train/test duplicate texts (reported, not removed) | All models must report this fingerprint (R3); duplicates are kept so we use the official split unchanged, and we disclose them on the data slide |
| D10 | Lexicon matcher: Damerau-Levenshtein (OSA) distance, d = 1 fixed by design; typo tolerance only for tokens NOT in the training vocabulary; K tuned on validation | Live chat has typos; swapped letters are the most common typo; only correcting non-words stops "aunt" matching "cunt" |
| D11 | tf-idf: word 1-2 grams, min_df=2, sublinear tf; LogisticRegression class_weight=balanced; C tuned on validation; all models train on TRAIN only, test is used only in Stage 7 | Standard strong text baseline; balanced weights for the ~1/3 offensive imbalance; one final test evaluation for all models (Hard Rule 2) |
| D12 | word2vec: skip-gram, 100-d, window 5, min_count 2, 20 epochs, trained on TRAIN tweets only, CRC32 hash + 1 worker for reproducibility. FFN: mean of word vectors, standardized with train stats -> Linear(100,h) -> ReLU -> Dropout 0.3 -> Linear(h,1); BCE loss with pos_weight; SGD + momentum 0.9, batch 64; hidden {64,128} x lr {0.05,0.1} and early stopping on validation macro-F1 | UC4-A1 asks for own word2vec feeding an FFN; Area A asks for SGD + backprop; averaging = simplest fixed-size input (loses word order: a known limitation) |
| D13 | DistilBERT (distilbert-base-uncased), raw text input, max_length 128, lr 2e-5, 3 epochs, batch 32, weight decay 0.01, 10% warmup, fp16 on GPU; best epoch by validation macro-F1; trained on Colab T4; weights uploaded to a public HF Hub repo `<hf-user>/swe015-distilbert-offensive`. Written for transformers 5.x (`processing_class=`, float `warmup_steps`; `tokenizer=` and `warmup_ratio` no longer exist) | Standard BERT fine-tuning recipe; raw text because WordPiece was pretrained on raw text; Hub hosting satisfies R7 (no weights in git) |
| D14 | Quantization: `torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)` on CPU; applied at load time from the FP32 Hub model (no extra hosted file); size = bytes of `torch.save(state_dict)` in MB (2^20), same method for FP32 and INT8; accuracy cost checked on validation | Dynamic quantization needs no calibration data or retraining, and Linear layers dominate Transformer CPU time; consistent size method makes FP32 vs INT8 comparable |
| D15 | Benchmark: test split used ONLY here; fingerprint asserted; 5 rows (rule, tf-idf, w2v+FFN, DistilBERT FP32, DistilBERT INT8); size = disk bytes of everything needed for inference (BERT: torch.save state_dict + config + tokenizer), MB = 2^20; latency = batch size 1, CPU, end-to-end, 10 warm-up + 200 timed runs over different test messages, mean (also median, p95); budget check 50 ms | R3/R4 exactly; per-message latency matches the chat use case; same size method for every model |
| D16 | All trained models live on the HF Hub repo (DistilBERT at the root, classical models under `classical/`); app.py downloads anything missing from models/ | R7: no weights in git, yet a fresh clone runs with one command |
| D17 | App: Gradio 6.32 Blocks; default model DistilBERT INT8 (quantized at start-up); dropdown for all 5 models; flagged messages masked as first letter + bullets; per-message latency log + session mean vs 50 ms budget; side-by-side table of all models on the last message; 5 warm-up calls per model at start-up | Meets every UC4 application requirement and R5; warm-up avoids a misleading slow first message |
| D18 | Benchmark run 1 showed INT8 slower than FP32 (oneDNN is the only quantized backend in this Windows PyTorch build; i7-8550U has no VNNI). Thread count is a deployment setting, so it is swept on VALIDATION messages (speed only, no labels) and one shared `CPU_THREADS` value is used for every model and the app. Run 1 is kept as results/results_run1_4threads.csv | Honest record: the change was prompted by run 1, the test set did not drive it, and both runs are kept |
| D19 | Thread sweep (validation messages, 100 runs each), mean ms FP32/INT8: 1 thread 62.43/42.10; 2: 38.67/45.16; 4: 37.27/61.00; 8: 45.19/94.81. Chosen CPU_THREADS = 1 for benchmark and app; full table shown on the slides | One-thread-per-message matches a CPU chat server serving many users; under it INT8 meets the 50 ms budget and FP32 does not. Showing the whole table (FP32 at 4 threads is fastest overall) keeps it honest |

## 9. Stage progress
| Stage | Description | Status |
|---|---|---|
| 0 | Requirements checklist | ☑ done (Q1 lexicon matcher = yes · Q2 budget = 50 ms · Q3 language = English, final) |
| 1 | Workspace and GitHub | ☑ done: venv ALL OK; first push = commit 08522dc |
| 2 | Data and splits | ☑ done: source cardiffnlp/tweet_eval/offensive; train 11,916 / val 1,324 / test 860; **test fingerprint `ac3947edadffd9ab`** |
| 3 | Regex normalization + lexicon matcher + tf-idf baseline | ☑ done: rule baseline K=200, val macro-F1 0.7135; tf-idf+LR C=4.0, val macro-F1 0.7007 (validation only; test scores come in Stage 7) |
| 4 | word2vec + FFN | ☑ done: hidden=128, lr=0.05, best epoch 3, val macro-F1 0.6412 (below both baselines: small-corpus word2vec + averaging; reported honestly) |
| 5 | DistilBERT fine-tuning on Colab | ☑ done: trained on Colab Tesla T4; best epoch 1 (checkpoint-373), val macro-F1 0.7754; model public at https://huggingface.co/mohdelsayed/swe015-distilbert-offensive |
| 6 | Quantization | ☑ done: 38/38 Linear layers -> dynamic INT8 (engine onednn); weights 255.5 MB -> 132.3 MB (1.93x); val macro-F1 FP32 0.7754 vs INT8 0.7809; same label on 88.9% of validation tweets |
| 7 | Benchmark + results.csv | ☑ done (final run 22 Nov 01:28, 1 CPU thread): test macro-F1 rule 0.7363 / tf-idf 0.7204 / w2v+FFN 0.6699 / DistilBERT FP32 0.7882 / INT8 0.8088; latency FP32 59.94 ms ❌, INT8 48.80 ms ✅ (p95 71.39); sizes 0.09 / 0.88 / 3.54 / 256.13 / 132.97 MB; run 1 (4 threads) kept |
| 8 | Gradio app | ☑ done: runs on the laptop with all 5 models, INT8 default, session mean 40.8 ms (within budget); classical models uploaded to the Hub `classical/` |
| 9 | README + fresh-env test | ☑ done: fresh clone + new venv + empty HF cache (HF_HOME) → `python app.py` downloaded all models and ran (22 Nov 16:33) |
| 10 | Architecture diagram | ☑ docs/architecture.png/.svg (drawn by docs/make_diagram.py), checked box by box against the code |
| 11 | Slide deck (PDF) | ◐ 12 slides (https://claude.ai/artifact/GdMbfnkboF56hx6jZLUKR1), video link on slide 1; PDF export pending |
| 12 | Video script + recording | ☑ recorded and narrated by Mohamed (2:31), sped up 1.27× to 1:59; YouTube unlisted: https://youtu.be/NUB3k0UknZ8 |
| 13 | Final submission audit | ◐ audit done in chat; PDF export + final push pending |
