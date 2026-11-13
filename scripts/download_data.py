import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/Junteng/Vision4Chart/resolve/main"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SOURCES = ["chartqa", "figureqa", "dvqa"]


def download(name):
    path = RAW_DIR / name
    if not path.exists():
        print(f"Downloading {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", path)
    return path


download("data/chart_mix_reproduce.csv")

for source in SOURCES:
    zip_path = download(f"data/vision4math_pic/{source}.zip")
    print(f"Extracting {source}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
