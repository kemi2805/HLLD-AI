#!/usr/bin/env python
"""The seed-ceiling logs (calea 78408, radii labelled %.2f, and 78415, the
small radii labelled %.0e) -> data/seed_ceiling.json for anatomy_figs.py
(2026-11-04).  In 78408 the radii 0, 1e-3 and 3e-3 all printed "0.00" and
overwrote one another, so its "0.00" rows are r = 3e-3 (the last written);
78415 holds r = 0, 3e-4, 1e-3.

    python parse_seed_ceiling.py LOG_78408 LOG_78415
"""
import json
import re
import sys

import numpy as np

GROUPS = ("first", "retry", "planar")
ROW = re.compile(r"^\s+(.+?)\s{2,}([\d.]+)% conv,\s+([\d.]+)% same\s+([\d.]+)% conv,\s+([\d.]+)% same\s+([\d.]+)% conv,\s+([\d.]+)% same")


def rows(path):
    out = []
    for line in open(path, errors="replace"):
        m = ROW.match(line)
        if m:
            v = [float(x) / 100 for x in m.groups()[1:]]
            out.append((m.group(1).strip(), v[0::2], v[1::2]))
    return out


def main():
    acc = {}
    net = None
    for path, coarse in ((sys.argv[1], True), (sys.argv[2], False)):
        for name, conv, same in rows(path):
            mm = re.match(r"exact \+ ([\d.e+-]+) step #\d", name)
            if mm:
                r = float(mm.group(1))
                if coarse and r == 0.0:
                    r = 3e-3                      # see the docstring
                if coarse and r < 3e-3:
                    continue
                acc.setdefault(r, []).append(conv)
            elif name.startswith("net ml_guess_r2d_it3") and net is None:
                net = conv
    radius = sorted(acc)
    conv = {g: [float(np.mean([c[i] for c in acc[r]])) for r in radius] for i, g in enumerate(GROUPS)}
    out = dict(radius=radius, conv=conv, net=dict(zip(GROUPS, net)),
               note="radius = the size of a random step from the known answer in the six seven-wave unknowns; "
                    "mean over the draws; one attempt, max_iter 40, accuracy 1e-8")
    json.dump(out, open("data/seed_ceiling.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
