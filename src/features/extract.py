import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.prepare import OUT_DIR, ROOT

RAW_DIR = ROOT / "data/raw"
FEATURES_DIR = ROOT / "data/features"
MODEL = "ViT-B-16-quickgelu"


def pad_square(image):
    size = max(image.size)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


class Charts(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(RAW_DIR / self.paths[i]).convert("RGB")
        return self.preprocess(pad_square(image))


@torch.no_grad()
def main():
    splits = pd.concat([pd.read_csv(OUT_DIR / f"{s}.csv") for s in ["train", "val", "test"]])
    images = sorted(splits.image.unique())
    statements = sorted(set(splits.option_0) | set(splits.option_1))

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained="openai", cache_dir=ROOT / "data/models"
    )
    model = model.cuda().eval()
    model.visual.output_tokens = True
    tokenizer = open_clip.get_tokenizer(MODEL)

    global_emb, patches = [], []
    loader = DataLoader(Charts(images, preprocess), batch_size=128, num_workers=8)
    for batch in tqdm(loader, desc="images"):
        pooled, tokens = model.visual(batch.cuda())
        global_emb.append(pooled.half().cpu())
        patches.append(tokens.half().cpu())

    text_emb = []
    for i in tqdm(range(0, len(statements), 512), desc="statements"):
        tokens = tokenizer(statements[i:i + 512]).cuda()
        text_emb.append(model.encode_text(tokens).half().cpu())

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"paths": images, "global": torch.cat(global_emb), "patches": torch.cat(patches)},
        FEATURES_DIR / "images.pt",
    )
    torch.save({"statements": statements, "emb": torch.cat(text_emb)}, FEATURES_DIR / "texts.pt")


if __name__ == "__main__":
    main()
