"""Acquire and parse the medical dataset (multi-label case study).

Downloads medical.arff and medical.xml from the canonical Mulan repository,
verifies both files against the frozen sha256 hashes recorded before any
experiment was run, and writes data/medical_undivided.npz with the exact
conventions used by build_register.py / build_register2.py (X float32,
Y int8, label names and order taken from the XML).

Usage: python get_medical.py
"""
import io, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
MULAN = "https://raw.githubusercontent.com/tsoumakas/mulan/master/data/multi-label/medical/"

# Frozen hashes (recorded at acquisition time, before any solve).
FROZEN = {
    "medical.arff": "ad0630a9a409ed0023c3bb56a0be7610b635c1ca185c6035e3b198b01167aac4",
    "medical.xml":  "51fa9d5c09e48caff3f0a0c75db1c9aa85944f8e35451f3ad8bd56fc422286ce",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_arff(path):
    """Dense or sparse ARFF -> (float32 matrix, attribute names)."""
    import arff  # liac-arff
    txt = Path(path).read_text(errors="replace")
    fmt = arff.DENSE
    for line in txt.splitlines():
        s = line.strip()
        if s and not s.lower().startswith(("@", "%")):
            if s.startswith("{"):
                fmt = arff.COO
            break
    d = arff.load(io.StringIO(txt), return_type=fmt)
    names = [a[0] for a in d["attributes"]]
    ncol = len(names)
    if fmt == arff.DENSE:
        rows = d["data"]
        M = np.zeros((len(rows), ncol), dtype=np.float32)
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                if v is None:
                    M[i, j] = np.nan
                else:
                    try:
                        M[i, j] = float(v)
                    except (ValueError, TypeError):
                        M[i, j] = 1.0 if str(v).upper() in ("1", "TRUE", "YES") else 0.0
    else:
        vals, ri, ci = d["data"]
        n = max(ri) + 1
        M = np.zeros((n, ncol), dtype=np.float32)
        for v, i, j in zip(vals, ri, ci):
            try:
                M[i, j] = float(v)
            except (ValueError, TypeError):
                M[i, j] = 1.0
    return M, names


def labels_from_xml(path):
    import re
    txt = Path(path).read_text(errors="replace")
    return re.findall(r'<label\s+name="([^"]+)"', txt)


def main():
    import requests
    DATA.mkdir(exist_ok=True)
    for fn, h in FROZEN.items():
        f = DATA / fn
        if not f.exists():
            r = requests.get(MULAN + fn, timeout=300)
            r.raise_for_status()
            f.write_bytes(r.content)
        got = sha(f)
        assert got == h, f"{fn}: sha256 mismatch\n  expected {h}\n  got      {got}"
        print(f"{fn}  sha256 OK  ({h[:16]}...)")

    M, names = parse_arff(DATA / "medical.arff")
    lab_names = labels_from_xml(DATA / "medical.xml")
    try:
        lab_idx = [names.index(l) for l in lab_names]
    except ValueError:  # Mulan convention: labels are the last L columns
        Lx = len(lab_names)
        lab_idx = list(range(len(names) - Lx, len(names)))
        assert set(np.unique(M[:, lab_idx])) <= {0.0, 1.0}, "label fallback not binary"
    feat_idx = [j for j in range(len(names)) if j not in set(lab_idx)]
    X = M[:, feat_idx].astype(np.float32)
    Y = (M[:, lab_idx] > 0.5).astype(np.int8)
    out = DATA / "medical_undivided.npz"
    np.savez_compressed(out, X=X, Y=Y,
                        feature_names=np.array([names[j] for j in feat_idx]),
                        label_names=np.array(lab_names))
    print(f"medical: n={X.shape[0]} p={X.shape[1]} L={Y.shape[1]} -> {out}")
    assert X.shape == (978, 1449) and Y.shape == (978, 45), "unexpected shapes"


if __name__ == "__main__":
    main()
