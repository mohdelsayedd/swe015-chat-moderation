"""Stage 5: fine-tune DistilBERT for offensive-message classification (Area C, UC4-C1).

Runs on Google Colab with a free T4 GPU (see notebooks/train_distilbert_colab.ipynb).
It also runs on CPU, but much more slowly (hours instead of minutes).

What "fine-tuning" means here:
    distilbert-base-uncased was pretrained by Hugging Face on a large amount of English
    text (masked-language modelling). We add a small classification head on top
    (Linear layer -> 2 classes) and train the whole network a little further on our
    11,916 training tweets. The pretrained layers already "understand" English; the
    short fine-tuning teaches them our task.

Why DistilBERT (UC4 suggests BERTurk / multilingual BERT):
    Our data is English, so we use an English encoder. DistilBERT is BERT distilled
    to 6 layers (~40% smaller, faster), which suits a CPU latency budget.

Input text: the RAW tweet (not our regex normalize()), because the model's own
WordPiece tokenizer was pretrained on raw text.

Model selection on VALIDATION only: we evaluate after every epoch and keep the
epoch with the best validation macro-F1 (load_best_model_at_end).

Usage (from the repo root):
    python -m src.bert_train                             # train, save to models/distilbert
    python -m src.bert_train --push USER/REPO_NAME       # ... and upload to the HF Hub
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from src.data import LABEL_NAMES, RESULTS_DIR, SEED, load_splits, set_seed

BASE_MODEL = "distilbert-base-uncased"
OUT_DIR = Path("models/distilbert")
MAX_LENGTH = 128        # tweets average ~23 words; 128 WordPiece tokens covers nearly all
LEARNING_RATE = 2e-5    # standard range for BERT fine-tuning (2e-5 to 5e-5)
EPOCHS = 3
BATCH_SIZE = 32


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: fine-tune DistilBERT")
    parser.add_argument("--model", default=BASE_MODEL, help="pretrained model to start from")
    parser.add_argument("--push", default=None, help="Hugging Face repo id, e.g. user/swe015-distilbert-offensive")
    parser.add_argument("--max-train", type=int, default=None, help="use only N training rows (smoke tests)")
    parser.add_argument("--epochs", type=float, default=EPOCHS)
    args = parser.parse_args()

    set_seed()
    splits = load_splits()
    train, val = splits["train"], splits["validation"]
    if args.max_train:
        train = train.sample(n=args.max_train, random_state=SEED)
    print(f"Device: {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Training rows: {len(train)}, validation rows: {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=LABEL_NAMES,                                   # {0: "clean", 1: "offensive"}
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )

    def encode(batch):
        # Truncate long tweets; padding is done per batch by the collator (faster).
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(train.reset_index(drop=True)).map(encode, batched=True)
    val_ds = Dataset.from_pandas(val.reset_index(drop=True)).map(encode, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")

    training_args = TrainingArguments(
        output_dir="checkpoints",
        learning_rate=LEARNING_RATE,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_steps=0.1,                  # in transformers 5.x a float = ratio of total steps
        eval_strategy="epoch",             # validate after every epoch ...
        save_strategy="epoch",
        load_best_model_at_end=True,       # ... and keep the best epoch
        metric_for_best_model="macro_f1",
        save_total_limit=1,
        logging_strategy="epoch",
        fp16=torch.cuda.is_available(),    # half precision on GPU: ~2x faster training
        seed=SEED,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,        # transformers 5.x name (was tokenizer=)
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Validation report for the chosen (best) epoch.
    logits = trainer.predict(val_ds).predictions
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(val["label"], preds, average="macro")
    print(f"\nBest-epoch validation macro-F1: {f1:.4f}")
    print(classification_report(val["label"], preds, target_names=list(LABEL_NAMES.values()), digits=4))
    print("Confusion matrix on validation (rows = true, cols = predicted) [clean, offensive]:")
    print(confusion_matrix(val["label"], preds))

    epoch_log = [{k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                 for row in trainer.state.log_history if "eval_macro_f1" in row or "loss" in row]
    summary = {
        "model": f"{BASE_MODEL} fine-tuned (FP32)",
        "chosen": {"learning_rate": LEARNING_RATE, "epochs_max": args.epochs,
                   "best_checkpoint": trainer.state.best_model_checkpoint,
                   "batch_size": BATCH_SIZE, "max_length": MAX_LENGTH},
        "val_macro_f1": round(f1, 4),
        "split_used_for_tuning": "validation",
        "trained_on": "GPU " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "train_rows": len(train),
        "log_history": epoch_log,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "val_distilbert.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))        # weights + config
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"\nSaved model to {OUT_DIR}/ and {RESULTS_DIR / 'val_distilbert.json'}")

    if args.push:
        # Public model repo: the app and the README download the weights from here (R7).
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(OUT_DIR), repo_id=args.push,
                          commit_message="Fine-tuned DistilBERT (SWE015 Stage 5)")
        api.upload_file(path_or_fileobj=str(RESULTS_DIR / "val_distilbert.json"),
                        path_in_repo="val_distilbert.json", repo_id=args.push)
        print(f"Uploaded to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
