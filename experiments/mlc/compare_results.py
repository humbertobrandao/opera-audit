"""EXACT comparison between two batteries (protocol v3, Section 9).
Frozen criterion: identical U_int, verdict, gcons_int and Jmin per
(label, regime, readout). The same J is not required (multiple optima).
Usage: compare_results.py A.json B.json
"""
import json, sys

A = json.load(open(sys.argv[1]))["labels"]
B = json.load(open(sys.argv[2]))["labels"]
FIELDS = ["U_int", "verdict", "gcons_int", "Jmin"]
bad = 0; tot = 0; jmin_cmp = 0; rl = 0; incoh = 0

def _tl(d):
    s = str(d.get("status", ""))
    # 'node limit exceeded' is a legacy status: in batteries generated before
    # the root-bound fix, it marks cells whose main solve hit the TL (the raw
    # root-node status used to clobber the sticky record).
    return ("time_limit" in s) or ("node limit" in s)

for k in A:
    for r in ("R1", "R2", "R3"):
        for rd in ("m05", "mtau"):
            key = f"{r}_{rd}"
            a, b = A[k][key], B[k][key]
            if _tl(a) or _tl(b):
                # resource-limited cell: values are bounds, not certificates.
                # It leaves the exact criterion; bound coherence is required:
                # the side that proved optimality must have U_int >= the
                # incumbent of the side that hit the TL (maximization).
                rl += 1
                ok = True
                if _tl(a) and not _tl(b):
                    ok = b["U_int"] >= a["U_int"]
                elif _tl(b) and not _tl(a):
                    ok = a["U_int"] >= b["U_int"]
                if not ok:
                    bad += 1; incoh += 1
                    print(f"BOUND INCOHERENCE k={k} {key}: U_int {a['U_int']} vs "
                          f"{b['U_int']} (status {a.get('status')} / {b.get('status')})")
                continue
            for f in FIELDS:
                if f not in a or f not in b:
                    continue          # core battery (--core): no Jmin
                tot += 1
                jmin_cmp += (f == "Jmin")
                if a[f] != b[f]:
                    bad += 1
                    print(f"DIVERGENCE k={k} {key} {f}: {a[f]} vs {b[f]}")
crit = "full (4 fields incl. Jmin)" if jmin_cmp else "core (U_int, verdict, gcons_int)"
suffix = ""
if rl:
    suffix = (f"; {rl} resource-limited cells excluded, bounds "
              + ("coherent" if incoh == 0 else f"INCOHERENT in {incoh}"))
print(f"{tot-bad}/{tot} fields identical "
      + ("=> EXACT MATCH" if bad == 0 else f"=> {bad} DIVERGENCES")
      + f"  [criterion: {crit}{suffix}]")
sys.exit(1 if bad else 0)
