"""Setup script: download model weights and install dependencies."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# Model download URLs
U2NET_URL = "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2net/u2net.pth"
U2NETP_URL = "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2netp/u2netp.pth"

MODELS_DIR = Path(__file__).resolve().parent / "models"


def download_file(url: str, dest: Path, description: str = ""):
    """Download a file with progress indication."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists")
        return

    print(f"  [DOWNLOAD] {description or dest.name} from {url}")
    try:
        urllib.request.urlretrieve(url, str(dest))
        print(f"  [DONE] Saved to {dest}")
    except Exception as e:
        print(f"  [ERROR] Failed to download: {e}")
        print(f"  Please manually download from: {url}")
        print(f"  And save to: {dest}")


def setup_models(download_u2net: bool = True, download_yolo: bool = True):
    """Download and set up model weights."""
    MODELS_DIR.mkdir(exist_ok=True)

    print("=== Setting up model weights ===")

    if download_u2net:
        print("\n--- U2-Net Saliency Detection ---")
        print("Note: U2-Net weights are ~176MB, U2-Net-lite (u2netp) is ~4.7MB")
        download_file(U2NETP_URL, MODELS_DIR / "u2netp.pth", "U2-Net-lite (u2netp)")
        # Full U2-Net is optional (large file)
        if "--full-u2net" in sys.argv:
            download_file(U2NET_URL, MODELS_DIR / "u2net.pth", "U2-Net (full)")

    if download_yolo:
        print("\n--- YOLOv8 Object Detection ---")
        print("YOLOv8-nano weights will be auto-downloaded on first use by ultralytics.")
        print("No manual download needed.")

    print("\n--- LAION Aesthetic Predictor ---")
    print("If using CLIP-based aesthetic scoring, install clip:")
    print("  pip install git+https://github.com/openai/CLIP.git")
    print("The aesthetic predictor weights need to be placed at:")
    print(f"  {MODELS_DIR / 'aesthetic_predictor.pth'}")
    print("If not available, the system will fall back to hand-crafted features.")

    print("\n=== Model setup complete ===")


def install_dependencies():
    """Install Python dependencies."""
    print("=== Installing dependencies ===")
    req_file = Path(__file__).resolve().parent / "requirements.txt"

    if req_file.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
    else:
        print("requirements.txt not found, installing core deps...")
        deps = [
            "numpy>=1.21.0",
            "opencv-python>=4.5.0",
            "PyYAML>=6.0",
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "ultralytics>=8.0.0",
            "flask>=2.3.0",
            "Pillow>=8.0.0",
            "tqdm>=4.60.0",
        ]
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)

    # Try to install CLIP (optional, may fail)
    print("\n--- Installing OpenAI CLIP (optional) ---")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/openai/CLIP.git"
        ])
        print("CLIP installed successfully.")
    except Exception as e:
        print(f"CLIP installation failed: {e}")
        print("The system will use fallback aesthetic scoring (hand-crafted features).")

    print("\n=== Dependencies installed ===")


def main():
    parser = argparse.ArgumentParser(description="AestheticCropper Setup")
    parser.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument("--download-models", action="store_true", help="Download model weights")
    parser.add_argument("--all", action="store_true", help="Install deps + download models")
    parser.add_argument("--full-u2net", action="store_true", help="Also download full U2-Net (176MB)")
    args = parser.parse_args()

    if args.all or args.install_deps:
        install_dependencies()

    if args.all or args.download_models:
        setup_models(download_u2net=True, download_yolo=True)

    if not (args.all or args.install_deps or args.download_models):
        parser.print_help()
        print("\nQuick start: python setup_project.py --all")


if __name__ == "__main__":
    main()
