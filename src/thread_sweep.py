"""Stage 7b: how does the number of CPU threads change DistilBERT latency (FP32 vs INT8)?

Why this exists:
    The first benchmark (4 threads, PyTorch's default on this laptop) showed the INT8
    model SLOWER than FP32 and above the 50 ms budget. On this Windows build the only
    quantization backend is oneDNN, and the i7-8550U has no VNNI (fast INT8) instructions.
    Thread count is a deployment setting, not a model setting, so we test 1/2/4/8 threads.

Fair-play rule: this uses VALIDATION messages only, and measures speed only (no labels),
so the test set still influences nothing. Whatever thread count is chosen is then used
for every model in the final benchmark and in the app.

Run from the repo root:
    python -m src.thread_sweep
Output: results/thread_sweep.csv
"""

import statistics
import time

import pandas as pd
import torch

from src.data import RESULTS_DIR, load_splits, set_seed
from src.quantize import BertClassifier, load_distilbert, quantize_int8

THREADS = [1, 2, 4, 8]
N_WARMUP, N_RUNS = 10, 100


def mean_latency_ms(model, messages) -> float:
    for msg in messages[:N_WARMUP]:
        model.predict_proba([msg])
    times = []
    for i in range(N_RUNS):
        start = time.perf_counter()
        model.predict_proba([messages[i % len(messages)]])
        times.append((time.perf_counter() - start) * 1000)
    return statistics.mean(times)


def main() -> None:
    set_seed()
    messages = load_splits()["validation"]["text"].tolist()
    fp32 = load_distilbert(quantized=False)
    int8 = BertClassifier(quantize_int8(fp32.model), fp32.tokenizer, "INT8")

    rows = []
    print("threads   FP32 ms   INT8 ms")
    for n in THREADS:
        torch.set_num_threads(n)
        row = {"threads": n,
               "fp32_ms": round(mean_latency_ms(fp32, messages), 2),
               "int8_ms": round(mean_latency_ms(int8, messages), 2)}
        rows.append(row)
        print(f"{n:>7} {row['fp32_ms']:>9} {row['int8_ms']:>9}")

    best = min(rows, key=lambda r: r["int8_ms"])
    print(f"\nFastest INT8 setting: {best['threads']} thread(s) -> {best['int8_ms']} ms "
          f"({'within' if best['int8_ms'] <= 50 else 'over'} the 50 ms budget)")
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "thread_sweep.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'thread_sweep.csv'}")


if __name__ == "__main__":
    main()
