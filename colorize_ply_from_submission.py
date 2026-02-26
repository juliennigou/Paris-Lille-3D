import numpy as np
from plyfile import PlyData, PlyElement

TEST_PLY = "test/MiniDijon9.ply"       # ou ton .ply train/test
SUBMISSION_TXT = "submission.txt"      # labels 1..6, 1 ligne par point
OUT_PLY = "colored_pro_segmentation.ply"

# Palette "pro segmentation" (lisible sur fond noir/gris dans CloudCompare)
# 1 ground, 2 buildings, 3 poles, 4 pedestrians, 5 cars, 6 vegetation
COLORS = {
    1: (160, 160, 160),  # ground: gris
    2: (220, 200, 160),  # buildings: beige clair
    3: (220, 60, 60),    # poles: rouge
    4: (170, 70, 200),   # pedestrians: violet
    5: (60, 140, 220),   # cars: bleu
    6: (60, 170, 90),    # vegetation: vert foncé
}

def main():
    ply = PlyData.read(TEST_PLY)
    v = ply["vertex"].data
    n_points = len(v)

    pred = np.loadtxt(SUBMISSION_TXT, dtype=np.int32)
    if len(pred) != n_points:
        raise ValueError(f"Mismatch: {n_points} points in PLY vs {len(pred)} labels in txt")

    if pred.min() < 1 or pred.max() > 6:
        raise ValueError(f"Expected labels in [1..6], got min={pred.min()} max={pred.max()}")

    rgb = np.zeros((n_points, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[pred == cls] = color

    vertex = np.empty(
        n_points,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )

    vertex["x"] = v["x"].astype(np.float32)
    vertex["y"] = v["y"].astype(np.float32)
    vertex["z"] = v["z"].astype(np.float32)
    vertex["red"] = rgb[:, 0]
    vertex["green"] = rgb[:, 1]
    vertex["blue"] = rgb[:, 2]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(OUT_PLY)
    print(f"✅ Wrote {OUT_PLY} (pro segmentation colors)")

if __name__ == "__main__":
    main()