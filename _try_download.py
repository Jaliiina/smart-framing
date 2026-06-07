"""Try alternative URLs for model weight downloads."""
import urllib.request
import socket
import os

socket.setdefaulttimeout(30)
os.makedirs("d:/cvProject/smart-framing/models", exist_ok=True)

# Try U2-Net-P weights
u2net_urls = [
    "https://huggingface.co/danielramez/u2net/resolve/main/saved_models/u2netp/u2netp.pth",
    "https://huggingface.co/spaces/danielramez/U2Net/resolve/main/saved_models/u2netp/u2netp.pth",
    "https://github.com/xuebinqin/U-2-Net/raw/master/saved_models/u2netp/u2netp.pth",
]

for url in u2net_urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        cl = resp.headers.get("Content-Length", "?")
        print(f"U2NET OK: {url} (size={cl})")
        # Download
        data = resp.read()
        path = "d:/cvProject/smart-framing/models/u2netp.pth"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path} ({len(data)} bytes)")
        resp.close()
        break
    except Exception as e:
        print(f"U2NET FAIL: {url} -> {e}")

# Try aesthetic predictor weights
aesthetic_urls = [
    "https://github.com/christophschuhmann/vqa-emnlp-2022-impl/raw/main/aesthetic-models/sac+logos+ava1-l14-linearMSE.pth",
    "https://github.com/christophschuhmann/vqa-emnlp-2022-impl/raw/master/aesthetic-models/sac+logos+ava1-l14-linearMSE.pth",
]

for url in aesthetic_urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30)
        cl = resp.headers.get("Content-Length", "?")
        print(f"AESTHETIC OK: {url} (size={cl})")
        data = resp.read()
        path = "d:/cvProject/smart-framing/models/aesthetic_predictor.pth"
        with open(path, "wb") as f:
            f.write(data)
        print(f"Saved: {path} ({len(data)} bytes)")
        resp.close()
        break
    except Exception as e:
        print(f"AESTHETIC FAIL: {url} -> {e}")
