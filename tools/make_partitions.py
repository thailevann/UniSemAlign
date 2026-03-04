import os
from pathlib import Path

IMG_DIR_CANDIDATES = ["images", "image"]
LAB_DIR_CANDIDATES = ["labels", "label", "masks", "mask"]

REPO_ROOT = Path(__file__).parent.parent

def get_repo_data_root(repo_root) -> Path:
    
    data_root = repo_root / "data"
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found at {data_root}")
    return data_root

def find_subdir(parent: Path, candidates: list[str]) -> Path:
    for name in candidates:
        p = parent / name
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find any of {candidates} under {parent}")

def list_images(img_dir: Path) -> list[str]:
    exts = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [p.name for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    if len(files) == 0:
        raise ValueError(f"No images found in {img_dir}")
    return files

def make_corrmatch_partitions(data_root: str, data_name: str, num_labeled: int):
    """
    data_root/
      crag/dataset_10/{train_sup,train_unsup,val,test}/{images,labels}/...
      glas/dataset_10/{...}
    Writes:
      partitions/{data_name}_{num_labeled}/labeled.txt, unlabeled.txt, val.txt
    Each line: "<rel_img_path> <rel_mask_path>"
    """
    data_root = Path(data_root)
    base = data_root / data_name / f"dataset_{num_labeled}"

    sup_split = base / "train_sup"
    unsup_split = base / "train_unsup"
    val_split = base / "val"

    sup_img_dir = find_subdir(sup_split, IMG_DIR_CANDIDATES)
    sup_lab_dir = find_subdir(sup_split, LAB_DIR_CANDIDATES)

    unsup_img_dir = find_subdir(unsup_split, IMG_DIR_CANDIDATES)
    unsup_lab_dir = find_subdir(unsup_split, LAB_DIR_CANDIDATES)

    val_img_dir = find_subdir(val_split, IMG_DIR_CANDIDATES)
    val_lab_dir = find_subdir(val_split, LAB_DIR_CANDIDATES)

    # output directory
    out_dir = REPO_ROOT / "partitions" / f"{data_name}_{num_labeled}"
    # out_dir = Path("partitions") / f"{data_name}_{num_labeled}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_pairs(img_dir: Path, lab_dir: Path, out_path: Path):
        names = list_images(img_dir)
        lines = []
        for fn in names:
            img_rel = (img_dir.relative_to(data_root)).as_posix() + "/" + fn
            lab_path = lab_dir / fn
            if not lab_path.exists():
                stem = Path(fn).stem
                found = None
                for cand in lab_dir.iterdir():
                    if cand.is_file() and cand.stem == stem:
                        found = cand
                        break
                if found is None:
                    raise FileNotFoundError(f"Missing label for {fn} in {lab_dir}")
                lab_rel = (found.relative_to(data_root)).as_posix()
            else:
                lab_rel = (lab_path.relative_to(data_root)).as_posix()

            lines.append(f"{img_rel} {lab_rel}")

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {len(lines)} lines -> {out_path}")

    write_pairs(sup_img_dir, sup_lab_dir, out_dir / "labeled.txt")
    write_pairs(unsup_img_dir, unsup_lab_dir, out_dir / "unlabeled.txt")
    write_pairs(val_img_dir, val_lab_dir, out_dir / "val.txt")

if __name__ == "__main__":
    data_root = get_repo_data_root(REPO_ROOT)
    
    # make_corrmatch_partitions("./data", "crag", 10)
    # make_corrmatch_partitions("./data", "crag", 20)
    make_corrmatch_partitions(str(data_root), "glas", 10)
    make_corrmatch_partitions(str(data_root), "glas", 20)
