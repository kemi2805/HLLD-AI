"""Print the tables of a scripts/pstar_evaluate.py report.

    python scripts/pstar_eval_table.py results/pstar_eval/eval.json [--subset all]
"""
import argparse
import json


def fmt(x, kind):
    if x is None:
        return "-"
    if kind == "e":
        return f"{x:.2e}"
    if kind == "p":
        return f"{100 * x:.2f}%"
    if kind == "f":
        return f"{x:.2f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--subset", action="append", default=None,
                    help="subset name(s) to print (default: all of them)")
    a = ap.parse_args()
    r = json.load(open(a.report))
    res = r["results"]
    models = [m["name"] for m in r["models"]]
    tsets = [t["name"] for t in r["testsets"]]
    cols = [("rows", "n_rows", "s"), ("HLLD valid", "hlld_valid_frac", "p"),
            ("rel med", "rel_median", "e"), ("rel p99", "rel_p99", "e"),
            ("rel max", "rel_max", "e"),
            ("valid (HLLD-valid rows)", "valid_frac_on_hlld_valid", "p"),
            ("valid (all rows)", "valid_frac_all_rows", "p"),
            ("same root", "ws_same_root", "p"), ("other root", "ws_other_root", "p"),
            ("ws no conv", "ws_not_converged", "p"),
            ("flux err med", "flux_err_median", "e"), ("flux err p99", "flux_err_p99", "e"),
            ("non-finite F", "ai_flux_nonfinite_all_rows", "s")]
    subsets = a.subset
    if subsets is None:
        subsets = []
        for v in res.values():
            for s in v:
                if s not in subsets:
                    subsets.append(s)
    for s in subsets:
        print(f"\n### subset: {s}\n")
        print("| test set | model | " + " | ".join(c[0] for c in cols) + " |")
        print("|" + "---|" * (len(cols) + 2))
        for t in tsets:
            for m in models:
                d = res.get(f"{m}|{t}", {}).get(s)
                if not d:
                    continue
                print(f"| {t} | {m} | " + " | ".join(fmt(d.get(k), kind)
                                                    for _, k, kind in cols) + " |")


if __name__ == "__main__":
    main()
