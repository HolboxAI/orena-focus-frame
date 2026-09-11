#!/usr/bin/env python3
"""ORena FOCUS scorer.

Given a predictions file with columns (id, prediction), compute:

  (a) OFFICIAL per-question correctness using focus.evaluation / focus.data.formats
      primitives.  Deterministic formats (binary, number, fo_class, time,
      percentage) are decided exactly as Evaluator._evaluate_single does:
          fmt.read(reference)  -> ValueError => incorrect
          fmt.read(prediction) -> ValueError => incorrect  (parse failure)
          fmt.compare(ref_parsed, pred_parsed)
      Judge formats (open_ended, matching, multiple_choice) are NOT decided here
      because they need an LLM judge; they are reported separately.  Their
      format-level verify() IS still applied, so we report how many would be
      rejected before the judge ever runs.

  (b) The 4-bucket PROXY mean.  The `ood` column is False on every row of the
      local data, so pre_evaluation_score() collapses to ID-only buckets.  We
      substitute ood_proxy = (procedure_type == "Sigmoid Resection"), which is
      the procedure that appears in test but never in train.  Buckets are
      (capability_group x ood_proxy), unweighted mean over populated buckets --
      identical arithmetic to Evaluator.pre_evaluation_score.

Because judge formats cannot be decided locally, the bucket mean is reported
three ways: judged rows EXCLUDED, judged rows forced WRONG (lower bound), and
judged rows forced RIGHT (upper bound).

Usage:
    python3 score.py PRED_FILE [--gt metadata/frames_test_true.parquet]
                               [--normalise] [--label NAME] [--json OUT.json]

PRED_FILE may be .csv / .jsonl / .json / .parquet and must have `id` and
`prediction` columns.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

from focus.data.formats import JUDGE_FORMATS, get_format_class
from focus.taxonomy import Capability

OOD_PROXY_PROCEDURE = "Sigmoid Resection"
DETERMINISTIC = ("binary", "number", "fo_class", "time", "percentage")


# ── format objects ────────────────────────────────────────────────────
def make_format(fmt_name: str):
    """Instantiate the official format class with default parameters."""
    cls = get_format_class(fmt_name)
    if fmt_name == "matching":            # needs a pattern; not present in test
        return cls(pattern=r".*")
    return cls()


# ── normalisation (routed by the TRUE answer_format column) ───────────
_CANON = ["Sponge", "Clip", "Specimen bag", "Silicone loop", "External drain",
          "Needle", "Gallstone", "Specimen", "Mesh", "Unknown foreign object"]
_CANON_LOWER = {c.lower(): c for c in _CANON}
_ALIAS = {"silicon loop": "Silicone loop", "silcon loop": "Silicone loop",
          "sillicon loop": "Silicone loop", "silicone loops": "Silicone loop",
          "clips": "Clip", "sponges": "Sponge", "needles": "Needle",
          "specimens": "Specimen", "gallstones": "Gallstone",
          "specimen bags": "Specimen bag", "external drains": "External drain",
          "specimen bag": "Specimen bag", "silicone loop": "Silicone loop",
          "no foreign objects": "none", "nothing": "none", "no": "none"}
_WORDNUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
            "ten": "10", "eleven": "11", "twelve": "12"}


def _canon_class(tok: str) -> str:
    t = re.sub(r"[^A-Za-z ]", " ", tok).strip().lower()
    t = re.sub(r"\s+", " ", t)
    if t in _ALIAS:
        return _ALIAS[t]
    if t in _CANON_LOWER:
        return _CANON_LOWER[t]
    for c in _CANON:
        if c.lower() == t.rstrip("s"):
            return c
    return tok.strip()


def normalise(text: str, fmt: str) -> str:
    """Best-effort repair of a raw generation, routed by the TRUE answer_format."""
    if text is None:
        return ""
    a = str(text).strip()
    # strip common chat scaffolding / thinking residue
    a = re.sub(r"<\|.*?\|>", "", a)
    a = re.sub(r"<think>.*?</think>", "", a, flags=re.S | re.I)
    a = re.sub(r"\s*\.?\s*Errors?\s*:.*$", "", a, flags=re.S | re.I)
    a = a.strip().strip('"').strip("'").strip()
    # take the first non-empty line for the short formats
    if fmt in ("binary", "number", "fo_class", "multiple_choice", "percentage", "time"):
        for ln in a.splitlines():
            if ln.strip():
                a = ln.strip()
                break
    a = a.rstrip(".").strip()

    if fmt == "binary":
        al = a.lower()
        m = re.search(r"\b(yes|no)\b", al)
        return m.group(1) if m else al
    if fmt == "number":
        al = a.lower()
        if al in _WORDNUM:
            return _WORDNUM[al]
        m = re.search(r"\d+", a)
        if m:
            return m.group(0)
        w = re.search(r"\b(" + "|".join(_WORDNUM) + r")\b", al)
        return _WORDNUM[w.group(1)] if w else a
    if fmt == "percentage":
        m = re.search(r"\d+(?:\.\d+)?", a)
        return m.group(0) if m else a
    if fmt == "time":
        ts = re.findall(r"\d{1,2}:\d{2}:\d{2}", a)
        return ", ".join(f"{int(t.split(':')[0]):02d}:{t.split(':')[1]}:{t.split(':')[2]}"
                         for t in ts) if ts else a
    if fmt == "multiple_choice":
        m = re.search(r"\b(top|bottom)\s*/\s*(left|right)\b", a, re.I)
        return f"{m.group(1).lower()}/{m.group(2).lower()}" if m else a.lower()
    if fmt == "fo_class":
        if re.fullmatch(r"(none|no foreign objects?|nothing|n/?a)\.?", a.strip(), re.I):
            return "none"
        parts = [_canon_class(p) for p in re.split(r"[,;]| and ", a) if p.strip()]
        parts = [p for p in parts if p]
        if parts and all(p == "none" for p in parts):
            return "none"
        parts = [p for p in parts if p != "none"]
        seen, out = set(), []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                out.append(p)
        return ", ".join(out) if out else a
    if fmt == "open_ended":
        return a[:300]
    return a


# ── core scoring ──────────────────────────────────────────────────────
def load_predictions(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".json"):
        df = pd.DataFrame(json.load(open(path)))
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"id", "prediction"} - set(df.columns)
    if missing:
        raise SystemExit(f"predictions file missing columns: {missing}")
    cols = ["id", "prediction"] + (["row"] if "row" in df.columns else [])
    return df[cols]


def score(gt: pd.DataFrame, preds: pd.DataFrame, do_normalise: bool) -> dict:
    use_row = "row" in preds.columns
    if use_row:
        pmap = dict(zip(preds["row"].astype(int), preds["prediction"]))
    else:
        gtd = int(gt["id"].astype(str).duplicated().sum())
        if gtd:
            print(f"WARNING: ground truth has {gtd} duplicate id(s) -- those rows "
                  f"cannot be keyed unambiguously by id. Supply a 'row' column.",
                  file=sys.stderr)
        pmap = dict(zip(preds["id"].astype(str), preds["prediction"]))
    fmts = {f: make_format(f) for f in gt["answer_format"].unique()}

    rows = []
    for _i, r in enumerate(gt.itertuples(index=False)):
        fmt_name = r.answer_format
        fmt = fmts[fmt_name]
        raw = pmap.get(_i) if use_row else pmap.get(str(r.id))
        text = "" if raw is None else str(raw)
        if do_normalise:
            text = normalise(text, fmt_name)

        # reference parse (Evaluator does this first; failure => incorrect)
        try:
            ref_parsed = fmt.read(str(r.answer))
            ref_ok = True
        except ValueError:
            ref_parsed, ref_ok = None, False

        # prediction parse / verify
        pred_ok, pred_parsed = True, None
        try:
            pred_parsed = fmt.read(text)
        except ValueError:
            pred_ok = False

        judged = fmt_name in JUDGE_FORMATS
        if not ref_ok or raw is None or not pred_ok:
            correct = False
        elif judged:
            correct = None                       # needs an LLM judge
        else:
            correct = bool(fmt.compare(ref_parsed, pred_parsed))

        rows.append({
            "row": _i,
            "id": str(r.id),
            "answer_format": fmt_name,
            "primary": r.primary_capability,
            "group": (Capability.from_any(r.primary_capability).group.value
                      if Capability.from_any(r.primary_capability) else r.primary_capability),
            "ood_proxy": bool(r.procedure_type == OOD_PROXY_PROCEDURE),
            "answer": str(r.answer),
            "prediction": text,
            "raw_prediction": "" if raw is None else str(raw),
            "ref_parse_ok": ref_ok,
            "pred_parse_ok": pred_ok,
            "missing": raw is None,
            "judged": judged,
            "correct": correct,
        })
    return {"results": pd.DataFrame(rows)}


REAL_GROUPS = ("object_recognition", "aggregation")


def bucket_mean(res: pd.DataFrame, judged_policy: str,
                only_real_groups: bool = False) -> tuple[float, pd.DataFrame]:
    """Unweighted mean over populated (group x ood_proxy) buckets.

    judged_policy: 'exclude' | 'wrong' | 'right'
    """
    df = res.copy()
    if only_real_groups:
        df = df[df["group"].isin(REAL_GROUPS)]
    if judged_policy == "exclude":
        df = df[~df["judged"]]
    elif judged_policy == "wrong":
        df["correct"] = [False if (c is None or pd.isna(c)) else c for c in df["correct"]]
    elif judged_policy == "right":
        df["correct"] = [
            (True if (j and c is None) else (False if c is None else c))
            for j, c in zip(df["judged"], df["correct"])
        ]
    df = df[[not (c is None or pd.isna(c)) for c in df["correct"]]]
    if df.empty:
        return float("nan"), pd.DataFrame()
    out = []
    for g in sorted(df["group"].unique()):
        for ood in (False, True):
            b = df[(df["group"] == g) & (df["ood_proxy"] == ood)]
            if b.empty:
                continue
            out.append({"group": g, "ood_proxy": ood,
                        "accuracy": float(b["correct"].astype(bool).mean()),
                        "count": len(b)})
    bdf = pd.DataFrame(out)
    return float(bdf["accuracy"].mean()), bdf


def report(res: pd.DataFrame, label: str) -> dict:
    n = len(res)
    det = res[~res["judged"]]
    jud = res[res["judged"]]
    lines = []
    A = lines.append
    A("=" * 78)
    A(f"  {label}   (n={n})")
    A("=" * 78)

    A("\n-- per-format --")
    A(f"{'format':<17}{'n':>6}{'parse_fail':>12}{'pf_rate':>9}{'correct':>9}{'acc':>8}   note")
    for f in sorted(res["answer_format"].unique()):
        s = res[res["answer_format"] == f]
        pf = int((~s["pred_parse_ok"]).sum())
        if f in JUDGE_FORMATS:
            A(f"{f:<17}{len(s):>6}{pf:>12}{100*pf/len(s):>8.2f}%{'-':>9}{'-':>8}   LLM judge required")
        else:
            c = int(s["correct"].astype(bool).sum())
            A(f"{f:<17}{len(s):>6}{pf:>12}{100*pf/len(s):>8.2f}%{c:>9}{100*c/len(s):>7.2f}%")
    A(f"{'-'*78}")
    dpf = int((~det["pred_parse_ok"]).sum())
    dc = int(det["correct"].astype(bool).sum())
    A(f"{'DETERMINISTIC':<17}{len(det):>6}{dpf:>12}{100*dpf/max(1,len(det)):>8.2f}%"
      f"{dc:>9}{100*dc/max(1,len(det)):>7.2f}%   <- micro accuracy")
    jpf = int((~jud["pred_parse_ok"]).sum())
    A(f"{'JUDGE-FORMATS':<17}{len(jud):>6}{jpf:>12}{100*jpf/max(1,len(jud)):>8.2f}%"
      f"{'?':>9}{'?':>8}   rejected pre-judge = auto-wrong")
    nref = int((~res["ref_parse_ok"]).sum())
    nmiss = int(res["missing"].sum())
    A(f"\nreference answers that fail official verify: {nref}  "
      f"(auto-incorrect for everyone)")
    A(f"predictions missing entirely: {nmiss}")

    out = {"label": label, "n": n,
           "deterministic_micro": dc / max(1, len(det)),
           "det_n": len(det), "det_correct": dc, "det_parse_fail": dpf,
           "judge_n": len(jud), "judge_parse_fail": jpf,
           "ref_parse_fail": nref, "missing": nmiss,
           "per_format": {}}
    for f in sorted(res["answer_format"].unique()):
        s = res[res["answer_format"] == f]
        e = {"n": len(s), "parse_fail": int((~s["pred_parse_ok"]).sum())}
        if f not in JUDGE_FORMATS:
            e["correct"] = int(s["correct"].astype(bool).sum())
            e["acc"] = e["correct"] / len(s)
        out["per_format"][f] = e

    A("\n-- 4-bucket proxy mean  (ood_proxy = procedure_type == 'Sigmoid Resection';")
    A("   restricted to object_recognition + aggregation, the 4 buckets populated")
    A("   on the real FRAME test set) --")
    for pol, desc in (("exclude", "judge-format rows EXCLUDED"),
                      ("wrong", "judge-format rows counted WRONG (lower bound)"),
                      ("right", "judge-format rows counted RIGHT (upper bound)")):
        sc, bdf = bucket_mean(res, pol, only_real_groups=True)
        A(f"\n  policy: {desc}")
        A(f"  {'group':<22}{'ood':>7}{'n':>7}{'acc':>9}")
        for b in bdf.itertuples(index=False):
            A(f"  {b.group:<22}{str(b.ood_proxy):>7}{b.count:>7}{100*b.accuracy:>8.2f}%")
        A(f"  {'BUCKET MEAN':<22}{'':>7}{len(bdf):>7}{100*sc:>8.2f}%")
        out[f"bucket_mean_{pol}"] = sc
        out[f"buckets_{pol}"] = bdf.to_dict("records")
    A("")
    print("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--gt", default="/home/ubuntu/orena/metadata/frames_test_true.parquet")
    ap.add_argument("--normalise", action="store_true",
                    help="apply the format-routed normaliser to raw predictions")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump", default=None, help="write per-row results to CSV")
    a = ap.parse_args()

    gt = pd.read_parquet(a.gt)
    preds = load_predictions(a.predictions)
    r = score(gt, preds, a.normalise)["results"]
    label = a.label or (f"{a.predictions}  [{'normalised' if a.normalise else 'raw'}]")
    out = report(r, label)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2, default=str)
    if a.dump:
        r.to_csv(a.dump, index=False)


if __name__ == "__main__":
    main()
