"""SDD -> x0.25 area down-sampling -> 256 tiles (stride 224) -> uint8 arrays per split."""
import numpy as np, cv2, os, sys
from concurrent.futures import ProcessPoolExecutor
SRC = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SDD_ROOT", "sdd_data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiles")
TILE, STRIDE, SCALE = 256, 224, 0.25
def origins(L):
    import math
    n = math.ceil((L - TILE) / STRIDE) + 1
    return sorted(set([i * STRIDE for i in range(n - 1)] + [L - TILE]))
def proc(i):
    img = cv2.imread(f"{SRC}/original_images/{i}.jpg")[:, :, ::-1]
    msk = cv2.imread(f"{SRC}/label_images_semantic/{i}.png", cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_AREA)
    msk = cv2.resize(msk, (int(w * SCALE), int(h * SCALE)), interpolation=cv2.INTER_NEAREST)
    T, M, C = [], [], []
    for y0 in origins(img.shape[0]):
        for x0 in origins(img.shape[1]):
            T.append(img[y0:y0+TILE, x0:x0+TILE]); M.append(msk[y0:y0+TILE, x0:x0+TILE]); C.append((int(i), y0, x0))
    return np.stack(T), np.stack(M), np.array(C)
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ids = sorted(f[:-4] for f in os.listdir(f"{SRC}/original_images"))
    assert len(ids) == 400
    splits = {"train": ids[:280], "val": ids[280:340], "test": ids[340:]}
    for name, idx in splits.items():
        with ProcessPoolExecutor(2) as ex:
            res = list(ex.map(proc, idx))
        np.save(f"{OUT}/{name}_img.npy", np.concatenate([r[0] for r in res]))
        np.save(f"{OUT}/{name}_mask.npy", np.concatenate([r[1] for r in res]))
        np.save(f"{OUT}/{name}_coords.npy", np.concatenate([r[2] for r in res]))
        print(name, sum(len(r[0]) for r in res), flush=True)
