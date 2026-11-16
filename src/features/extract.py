import argparse

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


def quadrants(image, size=224):
    """Resize the padded chart to 2*size and cut it into 4 tiles (row by row)."""
    image = pad_square(image).resize((2 * size, 2 * size), Image.BICUBIC)
    return [image.crop((x, y, x + size, y + size)) for y in (0, size) for x in (0, size)]


class Charts(torch.utils.data.Dataset):
    """Each chart as one 224x224 image, or (tiles=True) as its 4 quadrants at 448x448."""

    def __init__(self, paths, preprocess, tiles=False):
        self.paths = paths
        self.preprocess = preprocess
        self.tiles = tiles

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(RAW_DIR / self.paths[i]).convert("RGB")
        if self.tiles:
            return torch.stack([self.preprocess(tile) for tile in quadrants(image)])
        return self.preprocess(pad_square(image))


@torch.no_grad()
def extract_tiles(model, preprocess, images):
    """Patches of the 4 quadrants, concatenated: 4 x 196 = 784 patches per chart."""
    tile_global, patches = [], []
    loader = DataLoader(Charts(images, preprocess, tiles=True), batch_size=32, num_workers=8)
    for batch in tqdm(loader, desc="tiles"):
        pooled, tokens = model.visual(batch.flatten(0, 1).cuda())
        tile_global.append(pooled.half().cpu().unflatten(0, (-1, 4)))
        patches.append(tokens.half().cpu().unflatten(0, (-1, 4)).flatten(1, 2))
    return torch.cat(tile_global), torch.cat(patches)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", action="store_true",
                        help="also extract 2x2 quadrant patches at 448x448 (data/features/images_tiles.pt)")
    args = parser.parse_args()

    splits = pd.concat([pd.read_csv(OUT_DIR / f"{s}.csv") for s in ["train", "val", "test"]])
    images = sorted(splits.image.unique())
    statements = sorted(set(splits.option_0) | set(splits.option_1))

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained="openai", cache_dir=ROOT / "data/models"
    )
    model = model.cuda().eval()
    model.visual.output_tokens = True
    if args.tiles:
        full = torch.load(FEATURES_DIR / "images.pt")
        assert full["paths"] == images
        tile_global, patches = extract_tiles(model, preprocess, images)
        torch.save({"paths": images, "global": full["global"], "tile_global": tile_global, "patches": patches},
                   FEATURES_DIR / "images_tiles.pt")
        return
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
