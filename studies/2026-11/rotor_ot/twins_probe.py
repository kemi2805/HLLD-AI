"""How many solved rows of a harvest shard are bitwise / near duplicates?
    python3 twins_probe.py SHARD.npz [SHARD.npz ...]"""
import sys
import numpy as np

for f in sys.argv[1:]:
    z = np.load(f)
    UL, UR = z["U_L"].astype(float), z["U_R"].astype(float)
    a = np.ascontiguousarray(np.concatenate([UL, UR], axis=1))
    n = a.shape[0]
    nb = np.unique(a.view(np.dtype((np.void, 8 * a.shape[1]))).ravel()).size
    out = [f.split("/")[-1], n, nb]
    for d in (12, 10, 8, 6):
        q = np.round(a, d) + 0.0
        out.append(np.unique(q.view(np.dtype((np.void, 8 * q.shape[1]))).ravel()).size)
    print("%s rows %d  bitwise-unique %d  rounded 1e-12/-10/-8/-6: %s" % (out[0], out[1], out[2], out[3:]))
