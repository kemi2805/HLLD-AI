"""An explicit validation set for the trainer (2026-11-03; user: "fix the
validation split and retrain properly").

ml_guess._fit validated on a random --val-frac of the training file itself.
For the four-quadrant data that file holds the same problem many times (the
diagonal mirror twins are bit-identical in the canonical frame, the complex
faces are repeated x10, the ledger rows x20), so most validation rows had a
copy in training: val 0.0092 against train 0.0091, and the "best" checkpoint
was picked on memorisation.  This adds train_from_dataset(val_path=...) and
train_variant.py --val-data: the validation rows come from their own file
(a held-out time block), built by the same build_arrays_from_npz, and every
row of --data is trained on.  Without it nothing changes (bitwise).

Usage: python3 patch_val_split.py RMHD_FINAL_TREE   (at 9ca5a4b)
"""
import sys

root = sys.argv[1]


def sub(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70])
        s = s.replace(old, new)
    open(path, "w").write(s)


sub(f"{root}/rmhd/ml_guess.py", [
    # train_from_dataset: the argument ...
    ('''    n_candidates:  int = 1,
    **fit_kwargs,
) -> tuple[GuessMLP, Scaler]:''',
     '''    n_candidates:  int = 1,
    val_path:      Optional[str] = None,
    **fit_kwargs,
) -> tuple[GuessMLP, Scaler]:'''),
    # ... the validation arrays, built exactly as the training ones ...
    ('''    X, Y, Bx_range, Bx_full = build_arrays_from_npz(path, mirror=True,
                                                     verbose=verbose,
                                                     return_bx=True)
''',
     '''    X, Y, Bx_range, Bx_full = build_arrays_from_npz(path, mirror=True,
                                                     verbose=verbose,
                                                     return_bx=True)
    # `val_path`: validate on that file (a held-out time block, say) and
    # train on every row of `path`.  A random split of a file that holds one
    # problem several times (mirror twins, upweighted repeats) puts copies of
    # it on both sides, and the validation loss then measures memorisation.
    val_xy = (build_arrays_from_npz(val_path, mirror=True, verbose=verbose)[:2]
              if val_path is not None else None)
'''),
    # ... handed to _fit
    ('''                n_candidates=n_candidates, **fit_kwargs)''',
     '''                n_candidates=n_candidates, val_xy=val_xy, **fit_kwargs)'''),
    # _fit: the argument ...
    ('''    proxy_every:   int = 10,
) -> tuple[GuessMLP, Scaler]:''',
     '''    proxy_every:   int = 10,
    val_xy:        Optional[tuple] = None,
) -> tuple[GuessMLP, Scaler]:'''),
    # ... and the split it replaces
    ('''    rng = np.random.default_rng(seed)
    if mirror_paired:''',
     '''    rng = np.random.default_rng(seed)
    if val_xy is not None:                  # an explicit validation set
        i_tr = np.arange(len(X))
    elif mirror_paired:'''),
    ('''    X_va, Y_va = X[i_val], Y[i_val]''',
     '''    X_va, Y_va = val_xy if val_xy is not None else (X[i_val], Y[i_val])'''),
])

sub(f"{root}/scripts/train_variant.py", [
    ('''ap.add_argument("--val-frac", type=float, default=0.02)
''',
     '''ap.add_argument("--val-frac", type=float, default=0.02)
ap.add_argument("--val-data", default=None,
                help="validate on this dataset (e.g. a held-out time block) "
                     "and train on every row of --data; --val-frac is unused")
'''),
    ('''f"lr={a.lr}  seed={a.seed}  val_frac={a.val_frac}  device={a.device}", flush=True)''',
     '''f"lr={a.lr}  seed={a.seed}  "
      + (f"val_data={a.val_data}" if a.val_data else f"val_frac={a.val_frac}")
      + f"  device={a.device}", flush=True)'''),
    ('''    proxy_every=a.proxy_every, verbose=True)''',
     '''    proxy_every=a.proxy_every, val_path=a.val_data, verbose=True)'''),
])
print("patched", root)
