"""10 permutacoes R4 suplementares (protocolo v3, Secao 5): puro
pos-processamento por composicao dos oraculos R1-R3 ja resolvidos (nenhum
solve novo). Permutacao i derivada de default_rng([20260812, i]), i=1..10;
a principal (i=0) usa default_rng(20260812), identica a da bateria.
Uso: r4_supplement.py BATERIA.json SAIDA.json
"""
import json, sys
import numpy as np

bat = json.load(open(sys.argv[1]))["labels"]
labels = sorted(bat.keys(), key=int)
L = len(labels)
# ordem IDENTICA a R4_PROFILES da bateria: indice 0=(4,2), 1=(2,2), 2=(2,4)
SRC = ["R2", "R1", "R3"]
PROF = ["(4,2)", "(2,2)", "(2,4)"]
base = int(np.ceil(L / 3))
template = sum([[i] * base for i in range(3)], [])[:L]

def perm(i):
    rng = (np.random.default_rng(20260812) if i == 0
           else np.random.default_rng([20260812, i]))
    p = np.array(template); rng.shuffle(p)
    return p

rows = []
for i in range(11):
    p = perm(i)
    row = {"perm": i}
    for rd in ("m05", "mtau"):
        Us = [bat[k][f"{SRC[p[j]]}_{rd}"]["U"] for j, k in enumerate(labels)]
        gos = [bat[k][f"{SRC[p[j]]}_{rd}"]["verdict"] == "go"
               for j, k in enumerate(labels)]
        row[rd] = {"U_sum": float(np.sum(Us)), "U_bar": float(np.mean(Us)),
                   "GO": int(np.sum(gos)), "U_min": float(np.min(Us))}
    flips = sum(bat[k][f"{SRC[p[j]]}_m05"]["verdict"]
                != bat[k][f"{SRC[p[j]]}_mtau"]["verdict"]
                for j, k in enumerate(labels))
    row["flips_05_tau"] = int(flips)
    row["assign"] = {k: PROF[p[j]] for j, k in enumerate(labels)}
    if i == 0:                      # conferencia contra a bateria congelada
        ok = all(bat[k].get("R4_profile") == row["assign"][k] for k in labels
                 if "R4_profile" in bat[k])
        row["perm0_confere_com_bateria"] = bool(ok)
        assert ok, "main permutation does not match the battery one!"
    rows.append(row)
    print(f"perm {i:>2}: mtau GO {row['mtau']['GO']}/21 "
          f"Ubar={row['mtau']['U_bar']:6.2f} Umin={row['mtau']['U_min']:7.2f} | "
          f"m05 GO {row['m05']['GO']}/21 Ubar={row['m05']['U_bar']:6.2f} | "
          f"flips={row['flips_05_tau']}", flush=True)

json.dump({"perms": rows}, open(sys.argv[2], "w"), indent=1)
print("DONE", flush=True)
