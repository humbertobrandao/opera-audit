"""Derives every aggregated pre-certification number (Tables 14 and 15 and
their echoes in the text: 68.8%, 94.1%, 95.4%, 98.4%, 7373/308/4.2%,
3539/805/2734/77.3%, 3348/191/5.4%, 350, 2587/73.1%, residue 679) from the
per-dataset rows. Edit ROWS when adding datasets.

Fields per row: cells, scr_go, scr_nogo, srch_go, srch_nogo, indet,
UB_tot, Umon_tot, UB_ind, Umon_ind (the *_tot include the indeterminate
cells closed retrospectively; *_ind says how many were indeterminate).
"""
ROWS = {
    "medical":     (1890, 1832,  35,   0,   23,   0,   55,   58,  0,   0),
    "emotions":    ( 540,  480,   0,   3,   55,   2,    0,   15,  0,   0),
    "scene":       ( 108,  108,   0,   0,    0,   0,    0,    0,  0,   0),
    "enron":       ( 258,  141,  35,  11,   69,   2,   77,   94,  0,   0),
    "EukaryoteGO": ( 132,  129,   2,   0,    1,   0,    2,    3,  0,   0),
    "Mediamill":   ( 606,  504,  14,   8,   76,   4,   85,   90,  0,   0),
    "Delicious":   (5898, 2868, 482, 255, 1900, 393, 1979, 2540, 19, 252),
    "birds":       (  96,   88,   0,   1,    7,   0,    5,    7,  0,   0),
    "bibtex":      ( 954,  864,  50,   4,   36,   0,   83,   84,  0,   0),
    "yeast":       (  84,   24,   8,   4,   24,  24,   19,   45,  0,  15),
    "CAL500":      ( 876,   27, 179,  22,  543, 105,  301,  762,  0,  83),
}
def pct(a, b): return f"{100*a/b:.1f}%"
T = [sum(r[i] for r in ROWS.values()) for i in range(10)]
cells, sg, sn, hg, hn, ind, ub, um, ubi, umi = T
assert all(r[1]+r[2]+r[3]+r[4]+r[5] == r[0] for r in ROWS.values()), "row does not add up"
go, nogo = sg+hg, sn+hn
print(f"cells: {cells}")
print(f"basic screening: {sg+sn} ({pct(sg+sn,cells)})")
print(f"go: {go}; required the search: {hg} ({pct(hg,go)})")
print(f"no-go: {nogo}; only by the search: {hn} ({pct(hn,nogo)})")
print(f"U_mon closes {um-umi}/{nogo} ({pct(um-umi,nogo)}); beyond it: {nogo-(um-umi)} ({pct(nogo-(um-umi),nogo)}); +{umi} indet.")
print(f"U_B closes {ub-ubi} ({pct(ub-ubi,nogo)}); +{ubi} indet.")
print(f"candidates+U_mon: {sg+um}/{cells} ({pct(sg+um,cells)}); residue {cells-sg-um} = {hg} go + {nogo-(um-umi)} no-go + {ind-umi} indet.")
print(f"search alone: {pct(go+nogo,cells)}; search U hierarchy: {pct(cells-(ind-umi),cells)}")
