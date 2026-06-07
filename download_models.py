"""Download U2-Net and aesthetic predictor model weights.

Usage:
    python download_models.py           # Download all models
    python download_models.py --u2net   # Download U2-Net only
    python download_models.py --aesthetic  # Download aesthetic predictor only
"""

import argparse
import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


def download_file(url: str, path: str, desc: str):
    """Download a file with progress display."""
    print(f"Downloading {desc}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path)
        print(f"  -> Downloaded: {size:,} bytes to {path}")
        return True
    except Exception as e:
        print(f"  -> Failed: {e}")
        return False


def download_u2net():
    """Download U2-Net-P (lite) weights."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = str(MODELS_DIR / "u2netp.pth")

    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"U2-Net-P weights already exist at {path}")
        return True

    # Try multiple mirrors
    urls = [
        ("https://drive.google.com/uc?export=download&id=1rbSTGfoNq2GlJ2ZhoFKClQG2jHJ0W3yW",
         "Google Drive"),
        ("https://huggingface.co/spaces/danielramez/U2Net/resolve/main/saved_models/u2netp/u2netp.pth",
         "HuggingFace"),
    ]

    for url, source in urls:
        print(f"  Trying {source}...")
        if download_file(url, path, f"u2netp.pth from {source}"):
            return True

    print(f"\n  Automatic download failed. Please download manually:")
    print(f"  1. Visit: https://github.com/xuebinqin/U-2-Net#pre-trained-models")
    print(f"  2. Download u2netp.pth (4.7 MB)")
    print(f"  3. Place it at: {path}")
    return False


def download_aesthetic():
    """Download LAION aesthetic predictor weights (AVAClip-based)."""
    MODELS_DIR.mkdir(exist_ok=True)
    path = str(MODELS_DIR / "aesthetic_predictor.pth")

    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        print(f"Aesthetic predictor weights already exist at {path}")
        return True

    # The AVA+CLIP aesthetic predictor by christophschuhmann
    urls = [
        ("https://github.com/christophschuhmann/vqa-emnlp-2022-impl/raw/main/aesthetic-models/sac+logos+ava1-l14-linearMSE.pth",
         "GitHub (AVA-CLIP)"),
    ]

    for url, source in urls:
        print(f"  Trying {source}...")
        if download_file(url, path, f"aesthetic_predictor.pth from {source}"):
            return True

    print(f"\n  Automatic download failed. Please install CLIP and download weights manually:")
    print(f"  pip install openai-clip")
    print(f"  Place weights at: {path}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--u2net", action="store_true", help="Download U2-Net weights only")
    parser.add_argument("--aesthetic", action="store_true", help="Download aesthetic predictor only")
    args = parser.parse_args()

    if not args.u2net and not args.aesthetic:
        args.u2net = True
        args.aesthetic = True

    if args.u2net:
        download_u2net()
    if args.aesthetic:
        download_aesthetic()


if __name__ == "__main__":
    main()