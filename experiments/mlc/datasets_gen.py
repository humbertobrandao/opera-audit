"""Acquisition and parsing of the generality/scale datasets (frozen
protocol sha256 304b2e52..., 2026-08-27).

Source registry: 8 datasets from the canonical Mulan source (GitHub raw,
the same as medical); mediamill and EukaryoteGO are author-provided
(place the ARFFs in raw/<base>/), with the hash recorded at intake.
Corpus = union of all historically available partitions (protocol
rule 3b).

Usage: datasets_gen.py download   -> downloads the 8, records sha256
       datasets_gen.py parse      -> builds <base>_undivided.npz for raw/
       datasets_gen.py census     -> eligibility census (no solves)
"""
import sys, json, hashlib, io
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
RAW = HERE / "raw"; RAW.mkdir(exist_ok=True)
DATA = HERE / "data_gen"; DATA.mkdir(exist_ok=True)

MULAN = "https://raw.githubusercontent.com/tsoumakas/mulan/master/data/multi-label/"
REGISTRY = {
    # base: (arff files, xml file, stratum)
    "emotions":  (["emotions/emotions.arff"], "emotions/emotions.xml", "conf"),
    "scene":     (["scene/scene.arff"], "scene/scene.xml", "conf"),
    "yeast":     (["yeast/yeast.arff"], "yeast/yeast.xml", "conf"),
    "enron":     (["enron/enron.arff"], "enron/enron.xml", "conf"),
    "bibtex":    (["bibtex/bibtex.arff"], "bibtex/bibtex.xml", "conf"),
    "cal500":    (["cal500/cal500.arff"], "cal500/cal500.xml", "conf"),
    "birds":     (["birds/birds-train.arff", "birds/birds-test.arff"],
                  "birds/birds.xml", "conf"),
    "delicious": (["delicious/delicious.arff"], "delicious/delicious.xml", "escala"),
    # author-provided (place the ARFFs in raw/<base>/):
    "mediamill":   (None, None, "escala"),
    "eukaryotego": (None, None, "escala"),
}
K = 10  # protocol folds (eligibility P,N >= 10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download():
    import requests
    manifest = {}
    for base, (arffs, xml, _) in REGISTRY.items():
        if arffs is None:
            print(f"{base}: AUTHOR-PROVIDED (waiting for raw/{base}/)")
            continue
        d = RAW / base; d.mkdir(exist_ok=True)
        for rel in arffs + ([xml] if xml else []):
            fn = d / Path(rel).name
            if not fn.exists():
                r = requests.get(MULAN + rel, timeout=300)
                r.raise_for_status()
                fn.write_bytes(r.content)
            manifest[f"{base}/{fn.name}"] = sha(fn)
            print(f"{base}/{fn.name}  sha256 {manifest[f'{base}/{fn.name}'][:16]}...")
    (RAW / "manifest_download.json").write_text(json.dumps(manifest, indent=1))
    print("manifest saved")


def parse_arff(path):
    """Dense or sparse ARFF -> (float32 matrix, attribute names)."""
    import arff
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


def parse():
    meta = {}
    for base, (arffs, xml, estrato) in REGISTRY.items():
        d = RAW / base
        if not d.exists() or not list(d.glob("*.arff")):
            print(f"{base}: no ARFFs in raw/{base}, skipping"); continue
        files = sorted(d.glob("*.arff"))
        mats, names_ref = [], None
        for f in files:
            M, names = parse_arff(f)
            if names_ref is None:
                names_ref = names
            assert names == names_ref, f"attribute mismatch in {f}"
            mats.append(M)
        M = np.vstack(mats)                      # corpus = union (rule 3b)
        xmlf = list(d.glob("*.xml"))
        assert xmlf, f"{base}: label XML missing"
        lab_names = labels_from_xml(xmlf[0])

        def norm(s):
            return "".join(ch for ch in str(s).lower() if ch.isalnum())
        by_norm = {norm(nm): j for j, nm in enumerate(names_ref)}
        try:
            lab_idx = [names_ref.index(l) for l in lab_names]
        except ValueError:
            if all(norm(l) in by_norm for l in lab_names):
                lab_idx = [by_norm[norm(l)] for l in lab_names]
            else:            # Mulan fallback: labels = last L columns
                Lx = len(lab_names)
                lab_idx = list(range(len(names_ref) - Lx, len(names_ref)))
                tail = M[:, lab_idx]
                assert set(np.unique(tail)) <= {0.0, 1.0}, f"{base}: label fallback not binary"
        feat_idx = [j for j in range(len(names_ref)) if j not in set(lab_idx)]
        X = M[:, feat_idx]
        Y = (M[:, lab_idx] > 0.5).astype(np.int8)
        nan_ct = int(np.isnan(X).sum())
        np.savez_compressed(DATA / f"{base}_undivided.npz",
                            X=X, Y=Y,
                            feature_names=np.array([names_ref[j] for j in feat_idx]),
                            label_names=np.array(lab_names))
        meta[base] = {"n": int(X.shape[0]), "p": int(X.shape[1]),
                      "L_nominal": int(Y.shape[1]), "nan_features": nan_ct,
                      "arquivos": [f.name for f in files], "estrato": estrato,
                      "sha256_npz": sha(DATA / f"{base}_undivided.npz")}
        print(f"{base}: n={X.shape[0]} p={X.shape[1]} L={Y.shape[1]} nan={nan_ct}")
    (DATA / "meta_parse.json").write_text(json.dumps(meta, indent=1))


def census():
    out = {}
    for f in sorted(DATA.glob("*_undivided.npz")):
        base = f.name.replace("_undivided.npz", "")
        d = np.load(f, allow_pickle=True)
        X, Y = d["X"], d["Y"]
        n, p = X.shape; L = Y.shape[1]
        P = Y.sum(axis=0).astype(int)
        elig = [k for k in range(L) if P[k] >= K and (n - P[k]) >= K]
        cov = float(P[elig].sum() / max(P.sum(), 1))
        # attribute-type classification (frozen rule: >2 values)
        samp = X[: min(n, 2000)]
        distinct_gt2 = sum(1 for j in range(min(p, 4000))
                           if len(np.unique(samp[:, j][~np.isnan(samp[:, j])])) > 2)
        frac_cont = distinct_gt2 / min(p, 4000)
        tipo = "continuo" if frac_cont > 0.5 else "binario"
        out[base] = {"n": n, "p": p, "L_nominal": L, "L_eleg": len(elig),
                     "cobertura_positivos": round(cov, 4),
                     "cardinalidade": round(float(Y.sum(1).mean()), 2),
                     "tipo_atributo": tipo,
                     "frac_atributos_continuos": round(frac_cont, 3),
                     "elegiveis": elig,
                     "prevalencias": P.tolist()}
        print(f"{base:12s} n={n:6d} p={p:6d} L={L:4d} L_eleg={len(elig):4d} "
              f"cob={100*cov:5.1f}% card={out[base]['cardinalidade']:5.2f} tipo={tipo}")
    (DATA / "censo_elegibilidade.json").write_text(json.dumps(out, indent=1))
    print("census saved (recorded BEFORE any solve)")


def verify():
    m = json.loads((RAW / "manifest_download.json").read_text())
    bad = ok = 0
    for key, h in m.items():
        if key.startswith("_"):
            continue
        f = RAW / key
        if not f.exists():
            print("MISSING:", key); bad += 1; continue
        if sha(f) != h:
            print("HASH MISMATCH:", key); bad += 1
        else:
            ok += 1
    print(f"verify: {ok} ok, {bad} problems" + (" => INTEGRITY OK" if bad == 0 else " => FAILURE"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    {"download": download, "parse": parse, "census": census,
     "verify": verify}[sys.argv[1]]()
