"""Stage 7: final evaluation of ALL models on the SAME test set -> results/results.csv.

This is the only script that uses the test split (Hard Rule 2). Every number in the
results table on the slides comes from the CSV this script writes (Hard Rule 1, R4).

For each model it measures:
  macro_f1    on the 860 official test tweets (classes are imbalanced -> macro-F1)
  size_MB     disk size of everything the model needs to run (1 MB = 2^20 bytes)
              - rule baseline:  models/lexicon.json
              - tf-idf + LR:    models/tfidf_lr.joblib
              - word2vec + FFN: models/w2v.kv (+ .npy if any) + models/ffn.pt
              - DistilBERT:     torch.save(state_dict) of FP32 or INT8 weights
                                + config + tokenizer files (same method for both)
  latency_ms  mean time to classify ONE message (batch size 1, CPU only), end to end
              (preprocessing/tokenizing + model), averaged over N_RUNS >= 100 runs
              after warm-up. Median and 95th percentile are reported too.

It also checks the test-set fingerprint from Stage 2, so all rows provably used
the same test data (R3), and checks the 50 ms latency budget (UC4-D1, decision D3).

Run from the repo root (takes a few minutes):
    python -m src.benchmark
Output: results/results.csv, results/benchmark_meta.json, results/test_predictions.csv
"""

import datetime as dt
import io
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.baseline import MODEL_PATH as TFIDF_PATH, TfidfBaseline
from src.data import RESULTS_DIR, load_splits, set_seed, split_fingerprint
from src.env_info import hardware_summary
from src.lexicon import LEXICON_PATH, LexiconMatcher
from src.quantize import (BertClassifier, HF_MODEL_REPO, apply_cpu_threads, download_model,
                          load_distilbert, quantize_int8)
from src.w2v_ffn import FFN_PATH, W2V_PATH, W2VFFNModel

N_WARMUP = 10
N_RUNS = 200                 # requirement: >= 100 runs
LATENCY_BUDGET_MS = 50.0     # decision D3, fixed before measuring
BERT_EXTRA_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json"]


def files_mb(paths) -> float:
    return sum(Path(p).stat().st_size for p in paths) / 2**20


def bert_size_mb(model, model_path: Path) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    extra = files_mb(model_path / f for f in BERT_EXTRA_FILES if (model_path / f).exists())
    return buffer.getbuffer().nbytes / 2**20 + extra


def measure_latency(model, messages) -> dict:
    """Classify one message at a time; return mean / median / p95 in milliseconds."""
    for msg in messages[:N_WARMUP]:            # warm-up: first calls are slower (caches)
        model.predict_proba([msg])
    times = []
    for i in range(N_RUNS):
        msg = messages[i % len(messages)]      # different real messages, not one repeated
        start = time.perf_counter()
        model.predict_proba([msg])
        times.append((time.perf_counter() - start) * 1000)
    return {"mean": statistics.mean(times), "median": statistics.median(times),
            "p95": float(np.percentile(times, 95))}


def main() -> None:
    set_seed()
    apply_cpu_threads()                       # same thread setting as the app
    test = load_splits()["test"]
    texts, y = test["text"].tolist(), test["label"].to_numpy()

    # R3: prove this is the same test set as in Stage 2.
    expected = json.loads((RESULTS_DIR / "data_summary.json").read_text(encoding="utf-8"))
    expected_fp = expected["splits"]["test"]["fingerprint"]
    actual_fp = split_fingerprint(test)
    assert actual_fp == expected_fp, f"Test set changed! {actual_fp} != {expected_fp}"
    print(f"Test set: {len(test)} tweets, fingerprint {actual_fp} (matches Stage 2)")

    hw = hardware_summary()
    print(f"Hardware: {hw['cpu']}, {hw['logical_cores']} logical cores, {hw['ram']} RAM, "
          f"{hw['os']}, torch threads {torch.get_num_threads()}")

    # --- Load every model once ---
    bert_path = download_model()
    fp32 = load_distilbert(quantized=False, model_path=bert_path)
    int8 = BertClassifier(quantize_int8(fp32.model), fp32.tokenizer,
                          "DistilBERT INT8 (dynamic quantization)")
    w2v_files = [W2V_PATH, FFN_PATH] + sorted(Path(W2V_PATH).parent.glob(Path(W2V_PATH).name + "*.npy"))
    models = [
        ("Rule baseline (edit-distance lexicon)", "rules", LexiconMatcher.load(), files_mb([LEXICON_PATH])),
        ("tf-idf + logistic regression", "FP64", TfidfBaseline.load(), files_mb([TFIDF_PATH])),
        ("word2vec + feedforward net (own)", "FP32", W2VFFNModel.load(), files_mb(w2v_files)),
        ("DistilBERT fine-tuned", "FP32", fp32, bert_size_mb(fp32.model, bert_path)),
        ("DistilBERT fine-tuned, quantized", "INT8 (dynamic)", int8, bert_size_mb(int8.model, bert_path)),
    ]

    rows, preds_table = [], {"row": np.arange(len(test)), "label": y}
    for name, precision, model, size_mb in models:
        print(f"\n== {name} ==")
        start = time.perf_counter()
        probs = np.array(model.predict_proba(texts))
        pred = (probs >= 0.5).astype(int)
        print(f"  scored {len(texts)} test tweets in {time.perf_counter() - start:.1f} s")
        lat = measure_latency(model, texts)
        row = {
            "model": name,
            "precision": precision,
            "macro_f1": round(f1_score(y, pred, average="macro"), 4),
            "size_MB": round(size_mb, 2),
            "latency_ms": round(lat["mean"], 2),
            "latency_median_ms": round(lat["median"], 2),
            "latency_p95_ms": round(lat["p95"], 2),
            "latency_runs": N_RUNS,
            "meets_50ms_budget": lat["mean"] <= LATENCY_BUDGET_MS,
            "accuracy": round(accuracy_score(y, pred), 4),
            "offensive_recall": round(recall_score(y, pred, pos_label=1), 4),
            "test_fingerprint": actual_fp,
        }
        rows.append(row)
        preds_table[f"p_{precision.split()[0].lower()}_{len(rows)}"] = np.round(probs, 4)
        print("  " + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("model", "test_fingerprint")))

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "results.csv", index=False)
    pd.DataFrame(preds_table).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    meta = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "hardware": hw,
        "torch_threads": torch.get_num_threads(),
        "quantization_engine": torch.backends.quantized.engine,
        "latency": {"batch_size": 1, "warmup_runs": N_WARMUP, "timed_runs": N_RUNS,
                    "budget_ms": LATENCY_BUDGET_MS, "includes": "preprocessing/tokenization + model"},
        "size_unit": "MB = 2^20 bytes",
        "test_rows": len(test), "test_fingerprint": actual_fp,
        "distilbert_source": HF_MODEL_REPO,
    }
    (RESULTS_DIR / "benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n================ RESULTS (test set) ================")
    print(results[["model", "precision", "macro_f1", "size_MB", "latency_ms", "meets_50ms_budget"]]
          .to_string(index=False))

    # Failure examples for the limitations slide (P-5b): the app's model (INT8) wrong & confident.
    p_int8 = preds_table[list(preds_table)[-1]]
    wrong = np.where((p_int8 >= 0.5).astype(int) != y)[0]
    conf = np.abs(p_int8[wrong] - 0.5)
    print("\nMost confident INT8 mistakes on test (for the failure-example slide):")
    for i in wrong[np.argsort(-conf)][:6]:
        kind = "missed offensive" if y[i] == 1 else "false alarm"
        print(f"  row {i:>3} [{kind}, P(off)={p_int8[i]:.2f}] {texts[i][:120]}")

    print(f"\nSaved {RESULTS_DIR / 'results.csv'}, {RESULTS_DIR / 'benchmark_meta.json'}, "
          f"{RESULTS_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
