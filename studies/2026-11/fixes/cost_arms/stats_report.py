#!/usr/bin/env python3
"""tpatch2.py's statistics of one timing arm: the late rung's lanes (verdict,
stage, iterations, what answered them) and the in-process planar4 solves by
caller (calls, lanes, wall, iterations of the slowest lane).

    python3 stats_report.py DIR       (DIR = results/fixes/cost_arms/<arm>)"""
import collections, sys
import numpy as np
z = np.load(sys.argv[1] + "/rung_stats.npz", allow_pickle=True)
REASON = {0: "answered", 1: "not converged", 4: "unphysical", 5: "wave order", 6: "gate", 7: "ray"}
BY = {4: "planar seven-wave", 3: "four-unknown", 9: "other rung", 0: "lost"}
if "lane_c7_reason" in z.files:
    rs, st, it, by = z["lane_c7_reason"], z["lane_c7_stage"], z["lane_c7_n_iter"], z["lane_answered_by"]
    n_off = z["lane_n_offered_in_call"]
    print("lanes offered the late rung: %d (in calls with %s lanes offered: median %d, max %d)" % (
        rs.size, "1+", np.median(n_off), n_off.max()))
    print("  its verdict:", {REASON.get(int(k), k): int(v) for k, v in sorted(collections.Counter(rs.tolist()).items())})
    print("  answered by:", {BY.get(int(k), k): int(v) for k, v in sorted(collections.Counter(by.tolist()).items())})
    a = by == 4
    print("  its answers by stage:", dict(sorted(collections.Counter(st[a].tolist()).items())))
    for s in (1, 2, 3, 4):
        m = a & (st == s)
        if m.any():
            print("    stage %d: %d lanes, iterations (all stages summed) median %d, 90%% %d, max %d" % (
                s, m.sum(), np.median(it[m]), np.percentile(it[m], 90), it[m].max()))
    if (~a).any():
        print("  lanes it did not answer: iterations summed median %d, max %d" % (np.median(it[~a]), it[~a].max()))
if "call_who" in z.files:
    who, lanes, wall = z["call_who"], z["call_lanes"], z["call_wall"]
    mi, im, cv = z["call_max_iter"], z["call_iter_max"], z["call_converged"]
    print("\nin-process planar4 solves by caller:")
    print("  %-40s %6s %7s %8s %9s %9s %8s" % ("caller", "calls", "lanes", "wall s", "s/call", "cap", "it max"))
    for w in sorted(set(who.tolist()), key=lambda w: -wall[who == w].sum()):
        m = who == w
        print("  %-40s %6d %7d %8.1f %9.3f %9s %8.1f" % (w, m.sum(), lanes[m].sum(), wall[m].sum(),
              wall[m].mean(), "/".join(str(x) for x in sorted(set(mi[m].tolist()))), im[m].mean()))
