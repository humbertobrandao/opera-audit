"""Root gap on the budget-limited cells (Section 6.6), against the cell's
own incumbent, relative to |incumbent|: median 1.05 on Mediamill (748 cells
with a recorded root bound) and 2.96 on Delicious (4,168), against 0.00 on
the proven cells of both datasets.

Usage: analyze_budget_gaps.py records/battery_mediamill_seed*.json ...
(accepts several JSONs; aggregates per dataset)
"""
import json, sys, statistics as st, collections

acc = collections.defaultdict(lambda: {'tl': [], 'opt': []})
CEN = ('R1_m05','R1_mtau','R2_m05','R2_mtau','R3_m05','R3_mtau')
for f in sys.argv[1:]:
    d = json.load(open(f)); base = d['meta']['base']
    for cell in d['labels'].values():
        for key in CEN:
            c = cell.get(key)
            if not isinstance(c, dict) or c.get('root_bound') is None: continue
            inc = c.get('U_int')
            if inc is None: continue
            ratio = (c['root_bound'] - inc) / (abs(inc) if inc else 1)
            stt = str(c.get('status', ''))
            if stt.startswith('time_limit'): acc[base]['tl'].append(ratio)
            elif stt == 'optimal': acc[base]['opt'].append(ratio)
for base, v in acc.items():
    print(f"{base}: TL n={len(v['tl'])} median={st.median(v['tl']):.2f} | "
          f"proven n={len(v['opt'])} median={st.median(v['opt']):.2f}")
