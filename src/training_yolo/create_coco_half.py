import random
from pathlib import Path

def sample_half_txt(file_path, output_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Shuffle and take 50%
    random.seed(42)
    sampled = random.sample(lines, len(lines) // 2)

    with open(output_path, "w") as f:
        f.writelines(sampled)

    print(f"Wrote {len(sampled)} entries to {output_path}")

if __name__ == "__main__":
    base_path = Path("../datasets")  # match `path:` in coco.yaml

    sample_half_txt(base_path / "train2017.txt", base_path / "train2017_half.txt")
    sample_half_txt(base_path / "val2017.txt", base_path / "val2017_half.txt")