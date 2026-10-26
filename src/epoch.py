import shutil
from pathlib import Path

root = Path("/home/demirci/Downloads/silbe/quad_v1")
imgs = sorted((root / "train/images").glob("*.jpg"))
(root / "valid/images").mkdir(parents=True, exist_ok=True)
(root / "valid/labels").mkdir(parents=True, exist_ok=True)

for i, img in enumerate(imgs):
    if (i // 10) % 5 == 4:   # 10'luk blokların her 5'incisi valid'e
        lbl = root / "train/labels" / (img.stem + ".txt")
        shutil.move(str(img), root / "valid/images" / img.name)
        if lbl.exists():
            shutil.move(str(lbl), root / "valid/labels" / lbl.name)