#!/usr/bin/env python3
"""Full measured report for one model: 4 buckets, per-format, number error profile.

Input: the per-row dump written by `score.py --dump` (normalised run).
Output: a JSON blob + a printed table.
"""
import argparse, json, re, sys
import pandas as pd

def num(x):
    if x is None: return None
    m = re.search(r"-?\d+", str(x))
    return int(m.group(0)) if m else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("--label", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    d = pd.read_csv(a.dump, dtype=str, keep_default_na=False)
    d["ok"] = d["correct"].map({"True": True, "False": False})
    d["judged"] = d["judged"].map({"True": True, "False": False})
    out = {"label": a.label, "n": len(d)}

    real = d[d["group"].isin(["object_recognition", "aggregation"])]
    det = real[~real["judged"]]
    rows = []
    for g in ["aggregation", "object_recognition"]:
        for o in ["False", "True"]:
            b = det[(det["group"] == g) & (det["ood_proxy"].astype(str) == o)]
            if len(b) == 0: continue
            rows.append({"group": g, "ood": o == "True", "n": len(b),
                         "acc": float(b["ok"].mean())})
    out["buckets"] = rows
    out["bucket_mean"] = float(sum(r["acc"] for r in rows) / len(rows))

    pf = {}
    for f in sorted(d["answer_format"].unique()):
        s = d[d["answer_format"] == f]
        e = {"n": len(s), "parse_fail": int((s["pred_parse_ok"] == "False").sum())}
        if not bool(s["judged"].iloc[0]):
            e["acc"] = float(s["ok"].mean())
        pf[f] = e
    out["per_format"] = pf
    dd = d[~d["judged"]]
    out["det_micro"] = float(dd["ok"].mean()); out["det_n"] = len(dd)

    nb = d[d["answer_format"] == "number"].copy()
    nb["t"] = nb["answer"].map(num); nb["p"] = nb["prediction"].map(num)
    v = nb[nb["t"].notna()]
    parsed = v[v["p"].notna()]
    np_ = {"n": len(v), "unparseable": int(v["p"].isna().sum()),
           "exact": int((parsed["p"] == parsed["t"]).sum()),
           "under": int((parsed["p"] < parsed["t"]).sum()),
           "over": int((parsed["p"] > parsed["t"]).sum()),
           "mean_signed_err": float((parsed["p"] - parsed["t"]).mean()),
           "mae": float((parsed["p"] - parsed["t"]).abs().mean()),
           "mean_pred": float(parsed["p"].mean()),
           "mean_truth": float(v["t"].mean()),
           "pred_ge3_rate": float((parsed["p"] >= 3).mean()),
           "truth_ge3_rate": float((v["t"] >= 3).mean()),
           "pred_hist": {int(k): int(x) for k, x in parsed["p"].value_counts().items()},
           "truth_hist": {int(k): int(x) for k, x in v["t"].value_counts().items()},
           "recall_by_truth": {int(t): {"n": int(len(g)), "acc": float(g["ok"].mean())}
                               for t, g in v.groupby("t")}}
    out["number"] = np_

    print("=" * 74); print(f"  {a.label}   (n={len(d)})"); print("=" * 74)
    print(f"{'bucket':<34}{'n':>7}{'acc':>9}")
    for r in rows:
        print(f"  {r['group']:<22}ood={str(r['ood']):<6}{r['n']:>7}{100*r['acc']:>8.2f}%")
    print(f"  {'4-BUCKET MEAN':<30}{'':>7}{100*out['bucket_mean']:>8.2f}%")
    print(f"\n{'format':<18}{'n':>7}{'pf':>6}{'acc':>9}")
    for f, e in pf.items():
        print(f"{f:<18}{e['n']:>7}{e['parse_fail']:>6}" +
              (f"{100*e['acc']:>8.2f}%" if "acc" in e else f"{'judge':>9}"))
    print(f"{'DET MICRO':<18}{out['det_n']:>7}{'':>6}{100*out['det_micro']:>8.2f}%")
    n = np_
    print(f"\nnumber profile (n={n['n']}): exact {n['exact']} ({100*n['exact']/n['n']:.2f}%)  "
          f"under {n['under']} ({100*n['under']/n['n']:.2f}%)  over {n['over']} "
          f"({100*n['over']/n['n']:.2f}%)  unparseable {n['unparseable']}")
    print(f"  mean signed err {n['mean_signed_err']:+.3f}  MAE {n['mae']:.3f}  "
          f"mean pred {n['mean_pred']:.3f} vs truth {n['mean_truth']:.3f}  "
          f"pred>=3 {100*n['pred_ge3_rate']:.2f}% vs truth {100*n['truth_ge3_rate']:.2f}%")
    print("  recall by true count: " + "  ".join(
        f"{t}:{100*e['acc']:.1f}%(n={e['n']})" for t, e in sorted(n["recall_by_truth"].items()) if t <= 8))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
    return out

if __name__ == "__main__":
    main()
