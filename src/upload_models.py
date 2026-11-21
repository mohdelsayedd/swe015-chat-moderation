"""Upload the small classical models to the same Hugging Face repo as DistilBERT.

Why: R7 forbids model weights in the git repository, but the app must still run
from a fresh clone with one command. So every trained model lives on the Hub:
    <repo>/model.safetensors ...   DistilBERT (uploaded from Colab in Stage 5)
    <repo>/classical/...           rule lexicon, tf-idf + LR, word2vec + FFN (this script)
The app downloads whatever is missing from models/ on first start.

Needs a one-time login on this computer:  hf auth login
Run from the repo root:                   python -m src.upload_models
"""

from pathlib import Path

from huggingface_hub import HfApi

from src.quantize import HF_MODEL_REPO

MODEL_DIR = Path("models")
CLASSICAL_FILES = ["lexicon.json", "tfidf_lr.joblib", "w2v.kv", "ffn.pt"]


def main() -> None:
    files = [MODEL_DIR / name for name in CLASSICAL_FILES]
    files += sorted(MODEL_DIR.glob("w2v.kv*.npy"))      # gensim stores big arrays separately
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing {missing}: run src.lexicon, src.baseline and src.w2v_ffn first.")

    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")
    for f in files:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f"classical/{f.name}",
                        repo_id=HF_MODEL_REPO, commit_message=f"Add classical model file {f.name}")
        print(f"  uploaded {f.name} ({f.stat().st_size / 2**20:.2f} MB)")
    print(f"Done: https://huggingface.co/{HF_MODEL_REPO}/tree/main/classical")


if __name__ == "__main__":
    main()
