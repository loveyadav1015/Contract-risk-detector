"""One-time uploader for LegalBERT checkpoint to HuggingFace Hub.

Before running:
1) Replace YOUR_HF_USERNAME with your real HuggingFace username.
2) Run: huggingface-cli login
3) Run: python scripts/upload_legalbert.py
"""

from huggingface_hub import HfApi


def main() -> None:
    repo_id = "doflamingo1don/contract-risk-legalbert"
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False, repo_type="model")
    api.upload_folder(
        folder_path="models/best_model",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
