"""Frontier mechanics on the archived records (Sections 6.1/6.5/6.6).
Reproduces: (a) the 60-resolution cap reached in 857 scale cells (196 MM,
661 Del), 856 of which carry at least one resource-limited resolution;
(b) the cap on the panel in 12 cells (9 yeast, 1 bibtex, 2 cal500);
(c) the 38 CAL500 cases (all R3/mtau, nominal no-go, FP-free optimum,
U_int < 0), which support the reclassification as a finite frontier with
0 < rho* < 1.

Usage: analyze_frontier_records.py battery_*.json (all battery JSONs)
"""
import json, sys, collections

CEN = ('R1_m05','R1_mtau','R2_m05','R2_mtau','R3_m05','R3_mtau')
cap = collections.Counter(); cap_lim = collections.Counter()
deg = []
for f in sys.argv[1:]:
    d = json.load(open(f)); base = d['meta']['base']
    for lab, cell in d['labels'].items():
        for key in CEN:
            c = cell.get(key)
            if not isinstance(c, dict) or not isinstance(c.get('frontier'), dict): continue
            fr = c['frontier']; stt = str(c.get('status', ''))
            if fr.get('case') == 'finito' and fr.get('solves', 0) >= 60:
                cap[base] += 1
                if stt.startswith('time_limit') or 'limit' in stt: cap_lim[base] += 1
            if fr.get('case') == 'inf_degenerado':
                deg.append((base, lab, key, c['U_int'], c['fp'], stt, fr.get('rho')))
print("60-resolution cap per dataset:", dict(cap))
print("  with a resource-limited resolution:", dict(cap_lim))
print(f"inf_degenerado cases: {len(deg)}")
if deg:
    assert all(x[3] < 0 for x in deg), "there is a case with U_int >= 0"
    assert all(x[4] == 0 for x in deg), "there is a case with FP > 0"
    print("  all with U_int < 0 and fp = 0 (finite frontier, 0 < rho* < 1);",
          "scenarios:", collections.Counter(x[2] for x in deg))
