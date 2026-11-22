# Architecture diagram for SWE015 UC4: every box maps to code in the repo.
W, H = 1600, 900
INK, MUTED, LINE = "#1f2937", "#4b5563", "#6b7280"
BG_UI, BG_PRE, BG_MODEL, BG_KEY, BG_POST, BG_RUN = "#eef2ff", "#f1f5f9", "#ffffff", "#fff7ed", "#ecfdf5", "#f8fafc"
KEY = "#ea580c"
out = []
def rect(x, y, w, h, fill, stroke=LINE, sw=1.5, rx=10, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')
def text(x, y, s, size=15, weight=400, color=INK, anchor="start", family="Inter"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out.append(f'<text x="{x}" y="{y}" font-family="{family}, DejaVu Sans, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{s}</text>')
def box(x, y, w, h, title, lines, fill, stroke=LINE, sw=1.5, tcolor=INK, mono_last=None):
    rect(x, y, w, h, fill, stroke, sw)
    text(x + 12, y + 24, title, 16, 700, tcolor)
    for i, ln in enumerate(lines):
        mono = "`" in ln
        text(x + 12, y + 46 + 19 * i, ln.replace("`", ""), 13 if mono else 13.5, 400, MUTED,
             family="DejaVu Sans Mono" if mono else "Inter")
def arrow(x1, y1, x2, y2, color=LINE, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" marker-end="url(#ah)"{d}/>')
def path(dstr, color=LINE, sw=2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out.append(f'<path d="{dstr}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#ah)"{d}/>')

# Title
text(40, 44, "Architecture: real-time chat moderation (Use Case 4)", 26, 700)
text(40, 70, "Inference path at run time (top) · where each part runs and how the models were produced (bottom)", 15, 400, MUTED)

# Laptop frame (run time)
rect(24, 88, 1552, 540, "#ffffff", "#94a3b8", 1.5, 14, "6 5")
text(40, 112, "LAPTOP CPU: Intel i7-8550U · Windows 11 · PyTorch 2.15 CPU · CPU_THREADS = 1 (app and benchmark)", 13.5, 700, MUTED)

# 1 User + UI
box(40, 140, 200, 120, "Player (browser)", ["types a chat message", "e.g. \"stuuupid n00b\""], BG_UI)
box(40, 300, 200, 190, "Gradio UI · app.py", ["chat box + Message input", "Model dropdown", "  default: DistilBERT INT8", "Response-time log", "All-models table", "(Gradio 6.32, port 7860)"], BG_UI)
arrow(140, 260, 140, 296)

# Lane A: classical
box(300, 140, 250, 150, "Regex normalizer", ["`src/preprocess.py`", "HTML unescape, Turkish→ASCII,", "lowercase, drop URLs/@user,", "#tag→tag, leetspeak 1d10t→idiot,", "3+ repeated letters → 2"], BG_PRE)
box(610, 140, 300, 72, "Rule baseline · rules", ["lexicon K=200 (train) + Damerau-", ], BG_MODEL)
text(622, 204, "Levenshtein d=1 on non-words", 13.5, 400, MUTED)
box(610, 222, 300, 60, "tf-idf + logistic regr. · FP64", ["word 1–2-grams · C=4.0"], BG_MODEL)
box(610, 292, 300, 78, "word2vec + FFN · FP32", ["own skip-gram 100-d → mean →", "FFN 100→128→1 (SGD)"], BG_MODEL)

# Lane B: transformer
box(300, 400, 250, 110, "WordPiece tokenizer", ["raw text (no regex)", "`max_length = 128`", "distilbert-base-uncased vocab"], BG_PRE)
box(610, 390, 300, 64, "DistilBERT fine-tuned · FP32", ["6 layers + classifier head"], BG_MODEL)
rect(610, 464, 300, 82, BG_KEY, KEY, 2.5)
text(622, 488, "DistilBERT · INT8 (default)", 16, 700, KEY)
text(622, 510, "quantize_dynamic: 38 Linear → qint8", 13.5, 400, MUTED)
text(622, 529, "oneDNN · done at start-up", 13.5, 400, MUTED)

# arrows UI -> preprocessors
path("M240 350 C 270 350, 270 215, 296 215")
path("M240 430 C 270 430, 270 455, 296 455")
for y in (176, 252, 330):
    path(f"M550 215 C 580 215, 580 {y}, 606 {y}")
for y in (422, 505):
    path(f"M550 455 C 580 455, 580 {y}, 606 {y}")

# Post-processing
box(970, 140, 280, 110, "Decision", ["P(offensive) from the chosen model", "≥ 0.5 → offensive, else clean", "confidence = max(P, 1−P)", "rules: shows matched word"], BG_POST)
box(970, 270, 280, 90, "Masking", ["flagged message shown as", "\"y•• i••••\" (first letter + bullets)"], BG_POST)
box(970, 380, 280, 110, "Latency timer", ["time.perf_counter around", "predict_proba (one message)", "→ log row + session mean", "vs 50 ms budget"], BG_POST)
box(970, 510, 280, 92, "Side by side", ["all 5 models score the same", "message (not part of the timing)"], BG_POST)
for y in (176, 252, 330, 422, 505):
    path(f"M910 {y} C 940 {y}, 940 195, 966 195")
arrow(1110, 250, 1110, 266); arrow(1110, 360, 1110, 376); arrow(1110, 490, 1110, 506)

# Back to UI
box(1310, 140, 250, 170, "Shown in the UI", ["label: clean / offensive", "+ confidence + model name", "+ latency (ms)", "masked chat bubble", "log table + budget check"], BG_UI)
path("M1250 195 C 1280 195, 1280 200, 1306 200")
path("M1435 140 L 1435 128 L 190 128 L 190 136", LINE, 1.5, "5 4")
text(960, 124, "result rendered back in the browser", 12.5, 400, MUTED)
box(1310, 340, 250, 120, "Start-up (app.py)", ["download missing weights", "quantize FP32 → INT8", "5 warm-up calls per model", "apply CPU_THREADS = 1"], BG_RUN)

# Bottom: where it runs / how produced
rect(24, 648, 1552, 232, BG_RUN, "#94a3b8", 1.5, 14)
text(40, 674, "WHERE IT RUNS AND HOW THE MODELS WERE PRODUCED (offline)", 13.5, 700, MUTED)
box(40, 690, 280, 170, "Data · src/data.py", ["TweetEval offensive (HF Hub)", "official splits, seed 42:", "train 11,916 · val 1,324 · test 860", "test fingerprint", "`ac3947edadffd9ab`"], "#ffffff")
box(350, 690, 330, 170, "Training · laptop CPU", ["`src/lexicon.py` · `src/baseline.py`", "`src/w2v_ffn.py`", "tuned on validation only", "", "Training · Google Colab T4 GPU", "`src/bert_train.py` (3 epochs, best = 1)"], "#ffffff")
box(710, 690, 330, 170, "Hugging Face Hub (public)", ["mohdelsayed/", "  swe015-distilbert-offensive", "DistilBERT FP32 weights + tokenizer", "classical/: lexicon, tf-idf,", "  word2vec, FFN (4.7 MB)"], "#ffffff")
box(1070, 690, 490, 170, "Evaluation · laptop CPU (1 thread)", ["`src/quantize.py` · `src/thread_sweep.py` (validation)", "`src/benchmark.py` → results/results.csv", "test set used once, all 5 models, fingerprint checked", "macro-F1 · size (MB) · latency (mean of 200 runs)", "INT8: 0.8088 macro-F1 · 132.97 MB · 48.80 ms"], "#ffffff")
arrow(320, 775, 346, 775); arrow(680, 775, 706, 775); arrow(1040, 775, 1066, 775)
path("M860 690 C 860 650, 1435 650, 1435 464", "#94a3b8", 1.5, "5 4")
text(1120, 642, "downloaded at start-up (snapshot_download / hf_hub_download)", 12.5, 400, MUTED)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#ffffff"/>' + "".join(out) + "</svg>")
open("architecture.svg", "w", encoding="utf-8").write(svg)
