"""Stage 6: dynamic INT8 quantization of the fine-tuned DistilBERT (Area D, mandatory).

What dynamic quantization does:
    Every nn.Linear layer stores its weights as 32-bit floats (FP32). quantize_dynamic
    converts those weights to 8-bit integers (INT8) once, ahead of time; activations
    are quantized "dynamically" (on the fly) for each input. INT8 weights take ~4x
    less space and integer matrix multiplication is faster on a CPU. Embeddings and
    LayerNorm stay FP32, so the model shrinks by less than 4x overall.

Why dynamic (and not static) quantization:
    It needs no calibration data and no retraining, and Linear layers are where a
    Transformer spends most of its CPU time: a good fit for a one-day project.

Where the weights come from:
    The fine-tuned FP32 model is downloaded from our public Hugging Face Hub repo
    (Stage 5). The INT8 model is created from it here, on the CPU, in about a second.
    The app does the same at start-up, so no extra file has to be hosted (R7).

This script also checks, on VALIDATION (not test), how much accuracy quantization
costs. The final test scores, sizes and latencies are measured in Stage 7.

Run from the repo root:
    python -m src.quantize
Output: results/val_distilbert.json (downloaded), results/val_quantized.json,
        models/distilbert_int8_state.pt (only used to measure the INT8 size)
"""

import io
import json
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import RESULTS_DIR, load_splits, set_seed

HF_MODEL_REPO = "mohdelsayed/swe015-distilbert-offensive"   # created in Stage 5
MODEL_DIR = Path("models")
INT8_STATE_PATH = MODEL_DIR / "distilbert_int8_state.pt"
MAX_LENGTH = 128                                             # same as in training
# CPU threads used by the benchmark AND the app (one shared setting, so they match).
# Chosen with src/thread_sweep.py (Stage 7b, validation messages, speed only):
# with 1 thread INT8 was fastest (42.1 ms, within the 50 ms budget; FP32 62.4 ms).
# Deployment reasoning: a CPU chat server handles many messages at once -> one core each.
CPU_THREADS = 1
NEEDED_FILES = ["config.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "val_distilbert.json"]


def apply_cpu_threads() -> int:
    """Apply CPU_THREADS (if set) and return the thread count actually in use."""
    if CPU_THREADS:
        torch.set_num_threads(CPU_THREADS)
    return torch.get_num_threads()


def download_model(repo_id: str = HF_MODEL_REPO) -> Path:
    """Download (or reuse the cached copy of) the fine-tuned model from the Hub."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id, allow_patterns=NEEDED_FILES))


def quantize_int8(model: nn.Module) -> nn.Module:
    """Dynamic INT8 quantization of every nn.Linear layer."""
    with warnings.catch_warnings():
        # torch.ao.quantization is deprecated upstream (moving to torchao) but works
        # in our pinned torch 2.15.0 (decision D4); hide the expected warnings.
        warnings.simplefilter("ignore")
        from torch.ao.quantization import quantize_dynamic
        return quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def state_dict_mb(model: nn.Module) -> float:
    """Size of the model's weights when saved with torch.save, in MB (1 MB = 2^20 bytes).
    The same method is used for FP32 and INT8, so the two sizes are comparable."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 2**20


class BertClassifier:
    """DistilBERT (FP32 or INT8) with the same predict_proba interface as the other models."""

    def __init__(self, model, tokenizer, name: str):
        self.model, self.tokenizer, self.name = model.eval(), tokenizer, name

    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 32) -> list[float]:
        texts = list(texts)
        probs = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[start:start + batch_size], truncation=True,
                                 max_length=MAX_LENGTH, padding=True, return_tensors="pt")
            logits = self.model(**enc).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
        return probs

    def predict(self, texts) -> list[int]:
        return [int(p >= 0.5) for p in self.predict_proba(texts)]


def load_distilbert(quantized: bool, model_path: Path | None = None) -> BertClassifier:
    """Load the fine-tuned DistilBERT on CPU; optionally quantize it to INT8."""
    model_path = model_path or download_model()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
    if quantized:
        return BertClassifier(quantize_int8(model), tokenizer, "DistilBERT INT8 (dynamic quantization)")
    return BertClassifier(model, tokenizer, "DistilBERT FP32")


def count_layers(model: nn.Module) -> dict:
    counts = {}
    for module in model.modules():
        name = type(module).__module__ + "." + type(module).__name__
        if name.endswith((".Linear", ".Embedding", ".LayerNorm")):
            counts[name] = counts.get(name, 0) + 1
    return counts


def main(model_path: Path | None = None) -> None:
    set_seed()
    torch.set_num_threads(torch.get_num_threads())
    print(f"Quantization engine: {torch.backends.quantized.engine} "
          f"(supported: {torch.backends.quantized.supported_engines})")

    model_path = model_path or download_model()
    print(f"Model files: {model_path}")

    # Bring the Stage 5 training summary into the repo's results folder.
    val_src = model_path / "val_distilbert.json"
    if val_src.exists():
        RESULTS_DIR.mkdir(exist_ok=True)
        shutil.copy(val_src, RESULTS_DIR / "val_distilbert.json")
        print(f"Copied the Colab training summary to {RESULTS_DIR / 'val_distilbert.json'}")

    fp32 = load_distilbert(quantized=False, model_path=model_path)
    start = time.perf_counter()
    int8_model = quantize_int8(fp32.model)
    print(f"Quantization took {time.perf_counter() - start:.2f} s")
    int8 = BertClassifier(int8_model, fp32.tokenizer, "DistilBERT INT8 (dynamic quantization)")

    print("\nLayer types BEFORE quantization:")
    for name, n in count_layers(fp32.model).items():
        print(f"  {n:>3} x {name}")
    print("Layer types AFTER quantization:")
    for name, n in count_layers(int8.model).items():
        print(f"  {n:>3} x {name}")

    fp32_mb, int8_mb = state_dict_mb(fp32.model), state_dict_mb(int8.model)
    print(f"\nWeights size: FP32 {fp32_mb:.1f} MB -> INT8 {int8_mb:.1f} MB "
          f"({fp32_mb / int8_mb:.2f}x smaller)")

    # Accuracy cost of quantization, measured on VALIDATION only.
    val = load_splits()["validation"]
    print(f"\nScoring {len(val)} validation tweets with both models on CPU ...")
    p_fp32 = np.array(fp32.predict_proba(val["text"]))
    p_int8 = np.array(int8.predict_proba(val["text"]))
    y = val["label"].to_numpy()
    f1_fp32 = f1_score(y, (p_fp32 >= 0.5).astype(int), average="macro")
    f1_int8 = f1_score(y, (p_int8 >= 0.5).astype(int), average="macro")
    agree = float(np.mean((p_fp32 >= 0.5) == (p_int8 >= 0.5)))
    print(f"Validation macro-F1: FP32 {f1_fp32:.4f} | INT8 {f1_int8:.4f} "
          f"(change {f1_int8 - f1_fp32:+.4f})")
    print(f"FP32 and INT8 give the same label on {100 * agree:.1f}% of validation tweets")
    print(f"Mean |P(offensive) difference|: {np.mean(np.abs(p_fp32 - p_int8)):.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(int8.model.state_dict(), INT8_STATE_PATH)
    (RESULTS_DIR / "val_quantized.json").write_text(json.dumps({
        "method": "torch.ao.quantization.quantize_dynamic, {nn.Linear}, dtype=qint8",
        "engine": torch.backends.quantized.engine,
        "source_model": HF_MODEL_REPO,
        "weights_mb": {"fp32": round(fp32_mb, 1), "int8": round(int8_mb, 1)},
        "val_macro_f1": {"fp32": round(f1_fp32, 4), "int8": round(f1_int8, 4)},
        "val_label_agreement": round(agree, 4),
        "split_used": "validation",
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {RESULTS_DIR / 'val_quantized.json'} and {INT8_STATE_PATH}")


if __name__ == "__main__":
    main()
