"""Downloads of the pretrained checkpoints OCTCube pre-training starts from.

The RETFound repo on Hugging Face is gated: accept its terms on the model page once,
then provide a token with read access as `HF_TOKEN`, e.g. in the project's `.env`.

Usage, from the project root:

    uv run python -m octcubem.utils.checkpoints
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _PROJECT_ROOT / "ckpt"

# RETFound's OCT MAE weights, the initialisation OCTCube pre-training uses
_HF_REPO_ID = "YukunZhou/RETFound_mae_natureOCT"
_HF_FILENAME = "RETFound_mae_natureOCT.pth"


def download_retfound(hf_token: str | None = None, checkpoint_dir: str | Path = _CHECKPOINT_DIR) -> Path:
    """Downloads RETFound's OCT checkpoint into `checkpoint_dir` and returns its path.

    An existing, complete download is reused rather than fetched again. Pass the
    returned path to `main_pretrain_lightning.py` as `--resume`.
    """
    load_dotenv(_PROJECT_ROOT / ".env")
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN is not set. Add HF_TOKEN=<your token> to a .env file "
            "in the project root, or pass hf_token= explicitly."
        )
    ckpt_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_FILENAME,
        token=hf_token,
        local_dir=checkpoint_dir,
    )
    return Path(ckpt_path)


if __name__ == "__main__":
    print(download_retfound())
