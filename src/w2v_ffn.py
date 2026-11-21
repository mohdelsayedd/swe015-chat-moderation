"""Stage 4: our own neural network = word2vec (trained by us) + feedforward network.

Area A (feedforward network trained with SGD and backpropagation) and
Area B (word2vec embeddings), as required by UC4-A1:
    "your own word2vec trained on the training messages, feeding a feedforward network"

Pipeline for one message:
    text --normalize/tokenize--> words --word2vec--> one 100-d vector per word
         --average--> one 100-d vector for the message
         --standardize--> --FFN (100 -> hidden -> 1)--> sigmoid --> P(offensive)

Why average the word vectors? It is the simplest way to turn a variable-length
message into a fixed-size input that a feedforward network can take.
(Its weakness: word order is lost, so "not stupid" looks like "stupid".)

What we tune (on VALIDATION only): hidden layer size and learning rate.
Early stopping also uses validation macro-F1: we keep the epoch that scored best.

Run from the repo root:
    python -m src.w2v_ffn
Output: models/w2v.kv (+ .npy), models/ffn.pt, results/val_w2v_ffn.json,
        results/w2v_ffn_history.csv
"""

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gensim.models import KeyedVectors, Word2Vec
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn

from src.data import LABEL_NAMES, RESULTS_DIR, SEED, load_splits, set_seed
from src.preprocess import tokenize

MODEL_DIR = Path("models")
W2V_PATH = MODEL_DIR / "w2v.kv"
FFN_PATH = MODEL_DIR / "ffn.pt"

# --- word2vec settings ---
EMB_DIM = 100        # size of each word vector
WINDOW = 5           # how many neighbouring words count as "context"
MIN_COUNT = 2        # ignore words seen only once (no reliable vector for them)
W2V_EPOCHS = 20      # small corpus (~12k tweets) -> more passes help

# --- FFN settings ---
GRID_HIDDEN = [64, 128]
GRID_LR = [0.05, 0.1]
MAX_EPOCHS = 40
PATIENCE = 6         # stop if validation F1 hasn't improved for 6 epochs
BATCH_SIZE = 64
DROPOUT = 0.3
MOMENTUM = 0.9


def stable_hash(text: str) -> int:
    """gensim seeds each word's starting vector with hash(word). Python's built-in
    hash() changes every run, so we use CRC32 instead to make word2vec reproducible."""
    return zlib.crc32(text.encode("utf-8"))


def train_word2vec(token_lists) -> KeyedVectors:
    model = Word2Vec(
        sentences=token_lists,
        vector_size=EMB_DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,                 # skip-gram: predict context words from the centre word
        epochs=W2V_EPOCHS,
        seed=SEED,
        workers=1,            # 1 thread = deterministic result
        hashfxn=stable_hash,
    )
    return model.wv           # keep only the word vectors; the trainer isn't needed


def message_vectors(token_lists, wv: KeyedVectors) -> np.ndarray:
    """Average the word vectors of each message. Unknown words are skipped;
    a message with no known word gets the zero vector."""
    out = np.zeros((len(token_lists), wv.vector_size), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        known = [wv[t] for t in tokens if t in wv.key_to_index]
        if known:
            out[i] = np.mean(known, axis=0)
    return out


class FFN(nn.Module):
    """Feedforward network: input -> hidden (ReLU, dropout) -> 1 logit."""

    def __init__(self, in_dim: int, hidden: int, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),    # randomly zeroes units while training -> less overfitting
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # raw score (logit); sigmoid gives P(offensive)


def train_ffn(x_train, y_train, x_val, y_val, hidden: int, lr: float):
    """Train with mini-batch SGD + backpropagation; early-stop on validation macro-F1."""
    set_seed()
    model = FFN(x_train.shape[1], hidden)
    # pos_weight: an offensive example counts n_clean/n_off times as much in the loss,
    # same idea as class_weight="balanced" in the tf-idf model.
    n_off = float(y_train.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_train) - n_off) / n_off))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))
    xv = torch.from_numpy(x_val)
    gen = torch.Generator().manual_seed(SEED)

    best_f1, best_state, best_epoch, bad_epochs, history = -1.0, None, 0, 0, []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)       # shuffle each epoch
        total = 0.0
        for start in range(0, len(xt), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])        # forward pass
            loss.backward()                                 # backpropagation
            optimizer.step()                                # SGD update
            total += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_pred = (torch.sigmoid(model(xv)) >= 0.5).int().numpy()
        f1 = f1_score(y_val, val_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(total / len(xt), 4), "val_macro_f1": round(f1, 4)})

        if f1 > best_f1:
            best_f1, best_epoch, bad_epochs = f1, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_f1, best_epoch, history


class W2VFFNModel:
    """Same interface as the other models: predict_proba(list of texts)."""

    name = "word2vec + feedforward net (own)"

    def __init__(self, wv, ffn, mean, std):
        self.wv, self.ffn = wv, ffn.eval()
        self.mean, self.std = mean, std

    def predict_proba(self, texts) -> list[float]:
        x = message_vectors([tokenize(t) for t in texts], self.wv)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            return torch.sigmoid(self.ffn(torch.from_numpy(x))).tolist()

    def predict(self, texts) -> list[int]:
        return [int(p >= 0.5) for p in self.predict_proba(texts)]

    @classmethod
    def load(cls, w2v_path: Path = W2V_PATH, ffn_path: Path = FFN_PATH) -> "W2VFFNModel":
        wv = KeyedVectors.load(str(w2v_path))
        ckpt = torch.load(ffn_path, weights_only=False)
        ffn = FFN(ckpt["in_dim"], ckpt["hidden"])
        ffn.load_state_dict(ckpt["state_dict"])
        return cls(wv, ffn, ckpt["mean"], ckpt["std"])


def main() -> None:
    set_seed()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    splits = load_splits()
    train, val = splits["train"], splits["validation"]
    train_tok = [tokenize(t) for t in train["text"]]
    val_tok = [tokenize(t) for t in val["text"]]

    print("Training word2vec on the training tweets only ...")
    wv = train_word2vec(train_tok)
    print(f"word2vec vocabulary: {len(wv.key_to_index)} words, {EMB_DIM} dimensions")
    for probe in ["idiot", "stupid", "love", "game"]:
        if probe in wv.key_to_index:
            near = ", ".join(w for w, _ in wv.most_similar(probe, topn=5))
            print(f"  nearest to '{probe}': {near}")

    x_train = message_vectors(train_tok, wv)
    x_val = message_vectors(val_tok, wv)
    # Standardize each input dimension using TRAIN statistics only (helps SGD converge).
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    y_train = train["label"].to_numpy()
    y_val = val["label"].to_numpy()

    print("\nhidden     lr   best val macro-F1   best epoch")
    best = None
    for hidden in GRID_HIDDEN:
        for lr in GRID_LR:
            model, f1, epoch, history = train_ffn(x_train, y_train, x_val, y_val, hidden, lr)
            print(f"{hidden:>6} {lr:>6}   {f1:.4f}              {epoch}")
            if best is None or f1 > best[0]:
                best = (f1, hidden, lr, epoch, model, history)

    f1, hidden, lr, epoch, ffn, history = best
    print(f"\nChosen on validation: hidden={hidden}, lr={lr}, epoch={epoch} (val macro-F1 {f1:.4f})")

    MODEL_DIR.mkdir(exist_ok=True)
    wv.save(str(W2V_PATH))
    torch.save({"state_dict": ffn.state_dict(), "in_dim": EMB_DIM, "hidden": hidden,
                "mean": mean, "std": std}, FFN_PATH)

    final = W2VFFNModel(wv, ffn, mean, std)
    preds = final.predict(val["text"])
    print(classification_report(y_val, preds, target_names=list(LABEL_NAMES.values()), digits=4))
    print("Confusion matrix on validation (rows = true, cols = predicted) [clean, offensive]:")
    print(confusion_matrix(y_val, preds))

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "w2v_ffn_history.csv", index=False)
    (RESULTS_DIR / "val_w2v_ffn.json").write_text(json.dumps({
        "model": W2VFFNModel.name,
        "chosen": {"hidden": hidden, "lr": lr, "best_epoch": epoch, "optimizer": "SGD (momentum 0.9)",
                   "emb_dim": EMB_DIM, "dropout": DROPOUT, "batch_size": BATCH_SIZE},
        "val_macro_f1": round(f1, 4), "split_used_for_tuning": "validation",
        "w2v_vocabulary": len(wv.key_to_index),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {W2V_PATH}, {FFN_PATH}, {RESULTS_DIR / 'val_w2v_ffn.json'}")


if __name__ == "__main__":
    main()
