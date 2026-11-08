#!/usr/bin/env python3
"""Train the planar warm-start network on planar_data.py's arrays.

One MLP (14 -> 256 -> 256 -> 128 -> 8; gamma is the 14th input), the layout of rmhd.ml_guess.GuessMLP:
  4 regression outputs  ln p_LF, ln p_RF, By_CD, ln P_CD (standardised,
                        Huber loss);
  2 rotation logits     a rotation in front of the left / right slow wave
                        (binary cross-entropy, every row);
  2 steering logits     the left / right slow wave steered by the pressure
                        (binary cross-entropy, only rows a planar4-family
                        solver answered -- `steer_mask`).
Rows answered by the late rung (source 2), the ones the network is for, weigh
--hard-weight (default 5) in every loss, and the four-unknown solver's (source
3: the faces the late rung could not answer from the network's seed, 2% of
the hard rows in the pilot) --four-weight (default 20), or the steering head
learns the late rung's answer alone.

Split by time, per run and the same for every run, a row and its mirror
together: train t_frac < --val-from (0.7), validation up to --test-from
(0.8), test from there -- the 4Q at 64^2 and 128^2 are near-copies at equal
times, so a random split would leak.  Early stopping on the validation loss -- all three losses
(--stop-on all, v1), or the regression's alone (--stop-on reg): in v1 the
validation rotation loss rose from the first epochs while the regression was
still falling, so stopping on the sum stopped the seed's training early.

--baseline FILE (planar_baseline.py): production's seed on the same rows;
the report then gives its errors next to the network's, on the rows where it
is finite.

    python3 planar_train.py DATA.npz OUT.pt [--epochs 200] [--threads 32]
                            [--hidden 256,256,128] [--stop-on reg] [--baseline B.npz]
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

SOURCES = {0: "seven-wave", 1: "five-wave", 2: "late rung", 3: "four-unknown"}
TARGETS = ("lnp_LF", "lnp_RF", "By_CD", "lnP_CD")


class PlanarMLP(nn.Module):
    def __init__(self, n_in=14, hidden=(256, 256, 128), n_out=8):
        super().__init__()
        layers, w = [], n_in
        for h in hidden:
            layers += [nn.Linear(w, h), nn.SiLU()]
            w = h
        layers.append(nn.Linear(w, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def split_masks(t_frac, val_from, test_from):
    return t_frac < val_from, (t_frac >= val_from) & (t_frac < test_from), t_frac >= test_from


POS_W = {"rot": None, "steer": None}     # set in main: the heads' classes balanced


def losses(out, y, rot, steer, smask, w):
    reg = nn.functional.huber_loss(out[:, :4], y, reduction="none", delta=1.0).mean(dim=1)
    lr = nn.functional.binary_cross_entropy_with_logits(out[:, 4:6], rot, reduction="none",
                                                        pos_weight=POS_W["rot"]).mean(dim=1)
    ls = nn.functional.binary_cross_entropy_with_logits(out[:, 6:8], steer, reduction="none",
                                                        pos_weight=POS_W["steer"]).mean(dim=1)
    ls = ls * smask
    wsum = w.sum()
    return ((w * reg).sum() / wsum, (w * lr).sum() / wsum,
            (w * ls).sum() / torch.clamp((w * smask).sum(), min=1.0))


@torch.no_grad()
def evaluate(model, X, Y, ROT, STEER, SM, SRC, y_mean, y_std, names, YB=None, RB=None):
    """Per source: rotation and steering accuracy (and the base rate of the
    majority answer), the regression error in physical units (mean, median,
    90th percentile).  With a baseline (YB, RB: production's seed on the same
    rows, numpy), its median / 90th-percentile errors and the network's on the
    rows where it is finite, how often each is the closer of the two on all
    three signed-field unknowns (ln p_LF, ln p_RF, By_CD), and the accuracy
    of the rotations it suggests."""
    out = model(X)
    pr = out[:, :4] * y_std + y_mean
    yt = Y * y_std + y_mean
    rot_p = out[:, 4:6] > 0
    st_p = out[:, 6:8] > 0
    res = {}
    for s, name in names.items():
        m = SRC == s
        if not m.any():
            continue
        r = dict(rows=int(m.sum()))
        majority = lambda b: float(torch.bincount(b[:, 0].long() + 2 * b[:, 1].long(), minlength=4).max()) / b.shape[0]
        rt = ROT[m] > 0.5
        r["rot_acc"] = float(((rot_p[m] == rt).all(dim=1)).float().mean())
        r["rot_majority"] = majority(rt)             # always answering the commonest pair
        r["rot_any_rate"] = float(rt.any(dim=1).float().mean())
        if rt.any():
            r["rot_recall"] = float((rot_p[m] & rt).sum() / rt.sum())          # rotated sides found
            r["rot_false"] = float((rot_p[m] & ~rt).sum() / max(1, int((~rt).sum())))
        sm = SM[m] > 0.5
        if sm.any():
            stt = STEER[m][sm] > 0.5
            r["steer_acc"] = float(((st_p[m][sm] == stt).all(dim=1)).float().mean())
            r["steer_majority"] = majority(stt)
            r["steer_p_rate"] = float(stt.any(dim=1).float().mean())
            if stt.any():
                r["steer_recall"] = float((st_p[m][sm] & stt).sum() / stt.sum())
                r["steer_false"] = float((st_p[m][sm] & ~stt).sum() / max(1, int((~stt).sum())))
        err = (pr[m] - yt[m]).abs().numpy()
        for c, key in enumerate(TARGETS):
            r[key + "_mae"] = float(err[:, c].mean())
            r[key + "_med"] = float(np.median(err[:, c]))
            r[key + "_p90"] = float(np.quantile(err[:, c], 0.9))
        r["By_CD_scale"] = float(yt[m][:, 2].abs().mean())
        if YB is not None:
            mn = m.numpy()
            yb = YB[mn]
            ok = np.isfinite(yb).all(axis=1)
            r["base_rows"] = int(ok.sum())
            if ok.any():
                eb = np.abs(yb[ok] - yt[m].numpy()[ok])
                en = err[ok]
                for c, key in enumerate(TARGETS):
                    r[key + "_base_med"] = float(np.median(eb[:, c]))
                    r[key + "_base_p90"] = float(np.quantile(eb[:, c], 0.9))
                    r[key + "_net_med_same"] = float(np.median(en[:, c]))
                    r[key + "_net_p90_same"] = float(np.quantile(en[:, c], 0.9))
                sf = [0, 1, 2]
                r["net_closer_all3"] = float((en[:, sf] < eb[:, sf]).all(axis=1).mean())
                r["base_closer_all3"] = float((eb[:, sf] < en[:, sf]).all(axis=1).mean())
            if RB is not None:
                r["rot_base_acc"] = float((RB[mn] == rt.numpy()).all(axis=1).mean())
        res[name] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hard-weight", type=float, default=5.0)
    ap.add_argument("--four-weight", type=float, default=20.0)
    ap.add_argument("--cls-weight", type=float, default=1.0,
                    help="the classification losses' weight against the regression's")
    ap.add_argument("--val-from", type=float, default=0.7)
    ap.add_argument("--test-from", type=float, default=0.8)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=0, help="a pilot: subsample the training rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", default="256,256,128")
    ap.add_argument("--stop-on", choices=("all", "reg"), default="all")
    ap.add_argument("--baseline", default="", help="planar_baseline.py's output on DATA")
    a = ap.parse_args()
    hidden = tuple(int(h) for h in a.hidden.split(","))
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = np.load(a.data, allow_pickle=True)
    X, Y = d["X"].astype(np.float32), d["Y"].astype(np.float32)
    ROT, STEER = d["rot"].astype(np.float32), d["steer"].astype(np.float32)
    SM, SRC = d["steer_mask"].astype(np.float32), d["source"].astype(np.int64)
    tr, va, te = split_masks(d["t_frac"], a.val_from, a.test_from)
    if a.max_rows and tr.sum() > a.max_rows:
        keep = np.zeros(len(tr), bool)
        keep[rng.choice(np.flatnonzero(tr), a.max_rows, replace=False)] = True
        tr = keep
    print("rows: train %d, validation %d, test %d (runs %s)" % (
        tr.sum(), va.sum(), te.sum(), list(d["runs"])), flush=True)
    for name, m in (("train", tr), ("test", te)):
        print("  %s by source: %s" % (name, {SOURCES[int(s)]: int(((SRC == s) & m).sum())
                                              for s in np.unique(SRC)}), flush=True)
    x_mean, x_std = X[tr].mean(0), X[tr].std(0) + 1e-6
    y_mean, y_std = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    T = lambda v: torch.from_numpy(np.ascontiguousarray(v))
    Xn, Yn = T((X - x_mean) / x_std), T((Y - y_mean) / y_std)
    W = T(np.select([SRC == 2, SRC == 3], [a.hard_weight, a.four_weight], 1.0).astype(np.float32))
    ROTt, STt, SMt, SRCt = T(ROT), T(STEER), T(SM), T(SRC)
    y_mean_t, y_std_t = T(y_mean), T(y_std)
    # positives (a rotation, a pressure-steered wave) are the minority: weigh
    # them by negatives / positives over the weighted training rows, so the
    # heads are judged on balanced accuracy (a missed rotation or pressure
    # steering is the expensive mistake for the solver)
    for key, lab, msk in (("rot", ROT, np.ones(len(SRC), bool)), ("steer", STEER, SM > 0.5)):
        m = tr & msk
        pos = (lab[m] * W.numpy()[m][:, None]).sum(0)
        neg = ((1.0 - lab[m]) * W.numpy()[m][:, None]).sum(0)
        POS_W[key] = T(np.clip(neg / np.maximum(pos, 1e-9), 1.0, 100.0).astype(np.float32))
    print("pos_weight: rotation %s, steering %s" % (POS_W["rot"].numpy().round(2).tolist(),
                                                    POS_W["steer"].numpy().round(2).tolist()), flush=True)

    model = PlanarMLP(n_in=X.shape[1], hidden=hidden)
    print("network %d -> %s -> 8, %d parameters; early stop on %s" % (
        X.shape[1], hidden, sum(p.numel() for p in model.parameters()), a.stop_on), flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    itr, iva = np.flatnonzero(tr), np.flatnonzero(va)
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(itr)
        tot = np.zeros(3)
        for b in range(0, perm.size, a.batch):
            j = T(perm[b:b + a.batch])
            l_reg, l_rot, l_st = losses(model(Xn[j]), Yn[j], ROTt[j], STt[j], SMt[j], W[j])
            loss = l_reg + a.cls_weight * (l_rot + l_st)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += [float(l_reg.detach()), float(l_rot.detach()), float(l_st.detach())]
        sched.step()
        model.eval()
        with torch.no_grad():
            j = T(iva)
            v = losses(model(Xn[j]), Yn[j], ROTt[j], STt[j], SMt[j], W[j])
            vl = float(v[0]) if a.stop_on == "reg" else float(v[0] + a.cls_weight * (v[1] + v[2]))
        hist.append(dict(epoch=ep, train=(tot / max(1, -(-perm.size // a.batch))).tolist(),
                         val=[float(x) for x in v]))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print("epoch %3d: train reg/rot/steer %.4f %.4f %.4f  val %.4f %.4f %.4f%s  (%.0f s)" % (
                ep, *hist[-1]["train"], *hist[-1]["val"], "  *" if bad == 0 else "", time.time() - t0),
                flush=True)
        if bad >= a.patience:
            print("early stop at epoch %d" % ep, flush=True)
            break
    model.load_state_dict(best_state)
    model.eval()
    YB = RB = None
    if a.baseline:
        b = np.load(a.baseline, allow_pickle=True)
        YB = np.full((len(X), 4), np.nan)
        YB[b["idx"]] = np.where(b["ok"][:, None], b["Yb"], np.nan)
        RB = np.zeros((len(X), 2), bool)
        RB[b["idx"]] = b["rot"]
        print("baseline %s (%s): %d rows, %d finite" % (a.baseline, b["ckpt"], len(b["idx"]),
                                                       int(b["ok"].sum())), flush=True)
    rep = {}
    for name, m in (("validation", va), ("test", te)):
        j = T(np.flatnonzero(m))
        rep[name] = evaluate(model, Xn[j], Yn[j], ROTt[j], STt[j], SMt[j], SRCt[j], y_mean_t, y_std_t, SOURCES,
                             None if YB is None else YB[m], None if RB is None else RB[m])
        print("== %s" % name)
        for src, r in rep[name].items():
            print("  %-13s %s" % (src, ", ".join("%s %.4g" % (k, v) for k, v in r.items())))
    torch.save(dict(state_dict=model.state_dict(), x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                    n_in=X.shape[1], hidden=hidden, features="planar_data v1 (14: 13 canonical planar + gamma)",
                    targets="ln p_LF, ln p_RF, By_CD, ln P_CD | rot L R | steer P L R",
                    runs=list(d["runs"]), split=dict(val_from=a.val_from, test_from=a.test_from),
                    args=vars(a)), a.out)
    json.dump(dict(report=rep, history=hist, args=vars(a)), open(a.out.replace(".pt", "_report.json"), "w"),
              indent=1)
    print("saved %s (%.0f s)" % (a.out, time.time() - t0))


if __name__ == "__main__":
    main()
