"""Stage 8: real-time chat moderation demo (R1, R5, UC4 application requirements).

What the app does (Use Case 4 wording in brackets):
  - chat-like interface ["provide a chat-like interface"]
  - every message is labelled clean / offensive with a confidence, live
    ["each message is labeled live (clean / offensive) with confidence"]
  - offensive messages are masked in the chat ["flagged messages are masked"]
  - a side panel logs the response time of every message ["a side panel logs the
    response time of every message"]
  - a dropdown switches the model; default = quantized DistilBERT (R5)
    ["lets the user switch between models"]
  - the last message is also scored by all five models side by side (encouraged by R5)

Everything runs on the CPU. Run from the repo root:
    python app.py
Then open the address it prints (http://127.0.0.1:7860) if the browser doesn't open.
"""

import shutil
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download

from src.baseline import TfidfBaseline
from src.lexicon import LexiconMatcher
from src.quantize import (HF_MODEL_REPO, BertClassifier, apply_cpu_threads, load_distilbert,
                          quantize_int8)
from src.upload_models import CLASSICAL_FILES
from src.w2v_ffn import W2VFFNModel

MODEL_DIR = Path("models")
LATENCY_BUDGET_MS = 50.0                   # decision D3
DEFAULT_MODEL = "DistilBERT INT8 (quantized)"


def ensure_classical_models() -> None:
    """Fresh clone? Download the small classical model files from the Hub (R7)."""
    MODEL_DIR.mkdir(exist_ok=True)
    for name in CLASSICAL_FILES:
        if not (MODEL_DIR / name).exists():
            print(f"Downloading {name} from {HF_MODEL_REPO} ...")
            shutil.copy(hf_hub_download(HF_MODEL_REPO, f"classical/{name}"), MODEL_DIR / name)


def load_models() -> dict:
    ensure_classical_models()
    fp32 = load_distilbert(quantized=False)     # downloads from the Hub the first time
    int8 = BertClassifier(quantize_int8(fp32.model), fp32.tokenizer,
                          "DistilBERT INT8 (dynamic quantization)")
    return {                                     # display name -> (model, precision)
        DEFAULT_MODEL: (int8, "INT8"),
        "DistilBERT FP32": (fp32, "FP32"),
        "word2vec + feedforward net": (W2VFFNModel.load(), "FP32"),
        "tf-idf + logistic regression": (TfidfBaseline.load(), "FP64"),
        "Rule baseline (edit-distance lexicon)": (LexiconMatcher.load(), "rules"),
    }


def warm_up(models: dict) -> None:
    """The first call to a model is slow (memory allocation, caches). Do it now, at
    start-up, so the first real chat message shows its true latency (same idea as
    the 10 warm-up runs in benchmark.py)."""
    for model, _ in models.values():
        for _ in range(5):
            model.predict_proba(["warming up the model, good luck have fun"])


apply_cpu_threads()                           # same thread setting as the benchmark
print("Loading models (first start downloads ~270 MB from the Hugging Face Hub) ...")
MODELS = load_models()
warm_up(MODELS)
print(f"Ready. CPU threads: {torch.get_num_threads()}")


# Short names for the narrow side-panel tables, so the latency column stays visible.
SHORT_NAMES = {
    DEFAULT_MODEL: "BERT INT8",
    "DistilBERT FP32": "BERT FP32",
    "word2vec + feedforward net": "w2v+FFN",
    "tf-idf + logistic regression": "tf-idf+LR",
    "Rule baseline (edit-distance lexicon)": "rules",
}


def classify(model_name: str, message: str) -> dict:
    """Label one message with one model and time it (same timing as the benchmark)."""
    model, precision = MODELS[model_name]
    start = time.perf_counter()
    p_off = model.predict_proba([message])[0]
    latency_ms = (time.perf_counter() - start) * 1000
    offensive = p_off >= 0.5
    result = {
        "label": "offensive" if offensive else "clean",
        "confidence": p_off if offensive else 1 - p_off,
        "latency_ms": latency_ms,
        "precision": precision,
    }
    if isinstance(model, LexiconMatcher):        # rules give 0/1, so show the matched words
        result["matches"] = model.matches(message)
    return result


def mask(message: str) -> str:
    """Hide a flagged message but keep its shape: 'you idiot' -> 'y•• i••••'."""
    return " ".join(word[0] + "•" * (len(word) - 1) for word in message.split())


def respond(message, history, model_name, log):
    message = (message or "").strip()
    if not message:
        return "", history, log, gr.update(), gr.update()
    r = classify(model_name, message)
    flagged = r["label"] == "offensive"

    shown = f"🚫 {mask(message)}" if flagged else message
    verdict = (f"{'🔴 **offensive**' if flagged else '🟢 **clean**'} · confidence "
               f"{100 * r['confidence']:.0f}% · {model_name} ({r['precision']}) · "
               f"{r['latency_ms']:.1f} ms")
    if r.get("matches"):
        verdict += " · matched: " + ", ".join(f"{t}→{w}" for t, w in r["matches"])
    history = history + [{"role": "user", "content": shown},
                         {"role": "assistant", "content": verdict}]

    log = log + [[len(log) + 1, round(r["latency_ms"], 1), r["label"], round(r["confidence"], 2),
                  SHORT_NAMES[model_name]]]
    same_model = [row[1] for row in log if row[4] == SHORT_NAMES[model_name]]
    mean_ms = sum(same_model) / len(same_model)
    ok = "✅ within" if mean_ms <= LATENCY_BUDGET_MS else "❌ over"
    summary = (f"**{model_name}**: mean {mean_ms:.1f} ms over {len(same_model)} message(s) "
               f"in this session · {ok} the {LATENCY_BUDGET_MS:.0f} ms budget")

    # Side by side: every model's verdict on the same message (not part of the timed call).
    compare = []
    for name in MODELS:
        c = classify(name, message)
        compare.append([SHORT_NAMES[name], c["label"], round(c["confidence"], 2), round(c["latency_ms"], 1)])
    return "", history, log, summary, compare


def clear():
    return [], [], "", []


LOG_HEADERS = ["#", "latency (ms)", "label", "conf.", "model"]   # latency first: always visible
COMPARE_HEADERS = ["model", "label", "conf.", "latency (ms)"]

with gr.Blocks(title="SWE015 · Real-time chat moderation") as demo:
    gr.Markdown(
        "# 🎮 Real-time chat moderation\n"
        "Type a chat message. Each message is labelled **clean / offensive** on the CPU; "
        f"offensive messages are masked. Latency budget: **{LATENCY_BUDGET_MS:.0f} ms per message**. "
        "Default model: **DistilBERT, dynamically quantized to INT8**."
    )
    with gr.Row():
        with gr.Column(scale=3):
            model_choice = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model")
            chat = gr.Chatbot(value=[], label="Game chat", height=440)
            box = gr.Textbox(placeholder="Type a message and press Enter ...", label="Message",
                             submit_btn=True)
            clear_btn = gr.Button("Clear chat and log")
        with gr.Column(scale=2):
            gr.Markdown("### ⏱️ Response-time log")
            log_table = gr.Dataframe(value=[], headers=LOG_HEADERS, interactive=False,
                                     label="Every message, in order")
            budget = gr.Markdown("")
            gr.Markdown("### 🔍 All models on the last message")
            compare_table = gr.Dataframe(value=[], headers=COMPARE_HEADERS, interactive=False)

    log_state = gr.State([])
    box.submit(respond, [box, chat, model_choice, log_state],
               [box, chat, log_state, budget, compare_table])
    log_state.change(lambda rows: rows, log_state, log_table)
    clear_btn.click(clear, None, [chat, log_state, budget, compare_table])


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
