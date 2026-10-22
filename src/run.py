"""Whole pipeline: replay six stations with every policy, train the INT8 model, measure its cost on this CPU.

    python -m src.run     # writes results/results.json, docs/results.json, docs/series.json, README table

Every number in the README and on the demo page comes from this script.
"""
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort

from . import model, simulate as sim
from .fetch import ROOT, STATIONS, load

RESULTS = ROOT / "results" / "results.json"
TARGET = 0.95  # objective: no missed flood, and 95 % of alerts within 15 min, as with a measurement every 15 min
GRID = {"level": [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3], "rise": [0.4, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01],
        "model": [0.97, 0.95, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05]}
WINDOW = 18  # hours shown before and after the onset of each station's largest flood


def cpu_name() -> str:
    if sys.platform == "win32":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                           capture_output=True, text=True)
        return r.stdout.strip() or platform.processor()
    return platform.processor() or platform.machine()


def bench_us(fn, n: int = 500) -> float:
    for _ in range(20):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return float(np.median(t) * 1e6)


def pool(scores: list[dict]) -> dict:
    """Pooled figures over several stations: measurements relative to one per hour, alert delay, missed peak."""
    floods = [f for s in scores for f in s["floods"]]
    d = np.array([f["delay_min"] for f in floods if f["delay_min"] is not None], float)
    miss = np.array([f["peak_miss_cm"] for f in floods if f["peak_miss_cm"] is not None], float)
    return {"ratio": round(float(np.mean([s["per_year"] for s in scores]) / 8766), 3),
            "per_year": round(float(np.mean([s["per_year"] for s in scores]))),
            "floods": len(floods), "missed": len(floods) - len(d),
            "delay_mean_min": round(float(d.mean()), 1), "delay_p90_min": round(float(np.percentile(d, 90)), 1),
            "within_15_min": round(float((d <= 15).mean()), 3),
            "peak_miss_mean_cm": round(float(miss.mean()), 1), "peak_miss_max_cm": round(float(miss.max()), 1)}


def best(results: dict, family: str, stations: list[str]) -> float:
    """Operating point chosen on `stations` only: the cheapest setting that meets TARGET, else the closest to it."""
    rows = [(p, pool([results[family][p][s] for s in stations])) for p in GRID[family]]
    ok = [r for r in rows if r[1]["missed"] == 0 and r[1]["within_15_min"] >= TARGET]
    if ok:
        return min(ok, key=lambda r: r[1]["ratio"])[0]
    return min(rows, key=lambda r: (r[1]["missed"], -r[1]["within_15_min"], r[1]["ratio"]))[0]


def main() -> None:
    st = {}
    for code, name in STATIONS.items():
        t, h = load(code)
        t0, x = sim.to_grid(t, h)
        base, thr = sim.threshold(x)
        h1 = sim.hourly(x)
        st[code] = {"name": name, "t0": t0, "x": x, "base": base, "thr": thr, "h1": h1,
                    "floods": sim.events(x, thr), "X": sim.features(h1, base, thr, model.LAGS),
                    "Y": sim.target(x, thr, len(h1))}

    codes = list(st)
    results = {"fixed_60": {}, "fixed_15": {}, **{f: {p: {} for p in GRID[f]} for f in GRID}}
    for c, s in st.items():
        results["fixed_60"][c] = sim.score(s["x"], s["thr"], s["floods"], sim.fixed(s["h1"], False))
        results["fixed_15"][c] = sim.score(s["x"], s["thr"], s["floods"], sim.fixed(s["h1"], True))
        for p in GRID["level"]:
            results["level"][p][c] = sim.score(s["x"], s["thr"], s["floods"], sim.level(s["h1"], s["base"], s["thr"], p))
        for p in GRID["rise"]:
            results["rise"][p][c] = sim.score(s["x"], s["thr"], s["floods"], sim.rise(s["h1"], s["base"], s["thr"], p))

    # Leave one station out: the model that decides for station c never saw it. Its operating point is chosen on
    # the training stations (in-sample there), then applied to c.
    probs, p32 = {}, {}
    for c in codes:
        train = [o for o in codes if o != c]
        X = np.concatenate([st[o]["X"] for o in train])
        Y = np.concatenate([st[o]["Y"] for o in train])
        keep = ~np.isnan(X).any(1) & (X[:, 0] < 1)  # above the threshold the station is intensive anyway
        m = model.train(X[keep], Y[keep])
        fp32, int8 = model.export(m, X[keep], f"fold_{c}")
        s32, s8 = model.session(fp32), model.session(int8)
        probs[c] = {o: model.predict(s8, st[o]["X"]) for o in codes}  # INT8: what the station would run
        p32[c] = model.predict(s32, st[c]["X"])

    def model_flags(c: str, o: str, tau: float) -> np.ndarray:
        return sim.level(st[o]["h1"], st[o]["base"], st[o]["thr"], 1.0) | (probs[c][o] >= tau)

    in_sample = {c: {p: {o: sim.score(st[o]["x"], st[o]["thr"], st[o]["floods"], model_flags(c, o, p))
                         for o in codes if o != c} for p in GRID["model"]} for c in codes}
    for p in GRID["model"]:
        for c in codes:
            results["model"][p][c] = sim.score(st[c]["x"], st[c]["thr"], st[c]["floods"], model_flags(c, c, p))

    # Curves: every setting, pooled over the six stations (for the model, each station scored by its own fold).
    curves = {"fixed_60": pool(list(results["fixed_60"].values())), "fixed_15": pool(list(results["fixed_15"].values())),
              **{f: [{"param": p} | pool(list(results[f][p].values())) for p in GRID[f]] for f in GRID}}

    # Operating points chosen without the test station, then the test station's own figures pooled.
    chosen = {f: {} for f in GRID}
    for c in codes:
        others = [o for o in codes if o != c]
        for f in ("level", "rise"):
            chosen[f][c] = best(results, f, others)
        chosen["model"][c] = best({"model": {p: in_sample[c][p] for p in GRID["model"]}}, "model", others)
    operating = {"fixed_60": curves["fixed_60"], "fixed_15": curves["fixed_15"],
                 **{f: pool([results[f][chosen[f][c]][c] for c in codes]) | {"params": chosen[f]} for f in GRID}}

    agree = [float(((p32[c] >= chosen["model"][c]) == (probs[c][c] >= chosen["model"][c])).mean()) for c in codes]

    # Cost of one hourly decision on this CPU, single thread, with a model trained on all six stations.
    X = np.concatenate([st[c]["X"] for c in codes])
    Y = np.concatenate([st[c]["Y"] for c in codes])
    keep = ~np.isnan(X).any(1) & (X[:, 0] < 1)
    final = model.train(X[keep], Y[keep])
    fp32, int8 = model.export(final, X[keep], "final")
    sess = {"fp32": model.session(fp32), "int8": model.session(int8)}
    one = X[keep][:1]
    lin = [l for l in final if hasattr(l, "in_features")]
    cost = {**{f"{k}_us": round(bench_us(lambda v=v: v.run(None, {"levels": one})), 1) for k, v in sess.items()},
            "parameters": sum(p.numel() for p in final.parameters()), "macs": sum(l.in_features * l.out_features for l in lin),
            "fp32_bytes": fp32.stat().st_size, "int8_bytes": int8.stat().st_size,
            "int8_fp32_agreement": round(float(np.mean(agree)), 4)}

    stations, series = [], {}
    for c, s in st.items():
        valid = ~np.isnan(s["x"])
        first = datetime.fromtimestamp(s["t0"], timezone.utc)
        last = datetime.fromtimestamp(s["t0"] + sim.STEP * (len(s["x"]) - 1), timezone.utc)
        stations.append({"code": c, "name": s["name"], "from": first.date().isoformat(), "to": last.date().isoformat(),
                         "valid_share": round(float(valid.mean()), 3), "median_m": round(s["base"], 2),
                         "threshold_m": round(s["thr"], 2), "floods": len(s["floods"]),
                         "max_rise_1h_m": round(float(np.nanmax(s["x"][sim.PER_HOUR:] - s["x"][:-sim.PER_HOUR])), 2)})
        if not s["floods"]:
            continue
        on, end = max(s["floods"], key=lambda f: np.nanmax(s["x"][f[0]:f[1] + 1]))
        lo, hi = on - WINDOW * sim.PER_HOUR, min(on + WINDOW * sim.PER_HOUR, len(s["x"]) - 1)
        flags = {"fixed_60": sim.fixed(s["h1"], False), "fixed_15": sim.fixed(s["h1"], True),
                 "level": sim.level(s["h1"], s["base"], s["thr"], chosen["level"][c]),
                 "rise": sim.rise(s["h1"], s["base"], s["thr"], chosen["rise"][c]),
                 "model": model_flags(c, c, chosen["model"][c])}
        pts = {k: [int(i - lo) for i in sim.measured(v) if lo <= i <= hi] for k, v in flags.items()}
        this = {k: sim.score(s["x"], s["thr"], [(on, end)], v)["floods"][0] for k, v in flags.items()}
        series[c] = {"name": s["name"], "date": datetime.fromtimestamp(s["t0"] + sim.STEP * on, timezone.utc).isoformat()[:16],
                     "onset": on - lo, "threshold": round(s["thr"], 3),
                     "truth": [None if np.isnan(v) else round(float(v), 3) for v in s["x"][lo:hi + 1]], "measured": pts,
                     "flood": this}

    out = {"machine": {"cpu": cpu_name(), "os": platform.platform(), "onnxruntime": ort.__version__},
           "settings": {"target": TARGET, "quantile": sim.QUANTILE, "lags": model.LAGS, "grid": GRID,
                        "max_gap_h": sim.MAX_GAP // 3600, "merge_h": sim.MERGE // sim.PER_HOUR},
           "stations": stations, "curves": curves, "operating": operating, "cost": cost}
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(RESULTS, ROOT / "docs" / "results.json")
    (ROOT / "docs" / "series.json").write_text(json.dumps(series, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    readme = ROOT / "README.md"
    if readme.exists():
        text = re.sub(r"<!-- results -->.*<!-- /results -->", lambda _: f"<!-- results -->\n{table(out)}\n<!-- /results -->",
                      readme.read_text(encoding="utf-8"), flags=re.S)
        readme.write_text(text, encoding="utf-8")
    print(table(out))


POLICY_FR = {"fixed_60": "fixe, toutes les heures", "fixed_15": "fixe, tous les quarts d'heure",
             "level": "intensif au-dessus d'un niveau", "rise": "intensif si l'eau monte vite",
             "model": "intensif si le modèle INT8 prévoit le seuil"}


def fr(x: float) -> str:
    return f"{x:,}".replace(",", " ").replace(".", ",")


def table(out: dict) -> str:
    lines = ["| station | données | médiane (m) | seuil d'alerte (m) | crues | montée max en 1 h (m) |", "|---|---|---:|---:|---:|---:|"]
    lines += [f"| {s['name']} | {s['from']} → {s['to']} ({fr(round(100 * s['valid_share'], 1))} % valides) | {fr(s['median_m'])} | "
              f"{fr(s['threshold_m'])} | {s['floods']} | {fr(s['max_rise_1h_m'])} |" for s in out["stations"]]
    lines += ["", "Pour chaque station, le réglage est choisi sur les cinq autres : le moins cher qui ne manque aucune crue et donne "
              f"au moins {fr(round(100 * out['settings']['target']))} % des alertes en 15 min ou moins.", "",
              "| règle | mesures par an | × une par heure | crues manquées | retard moyen de l'alerte | 90 % des alertes en moins de | alertes en 15 min ou moins | pic manqué en moyenne | pic manqué au pire |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in out["operating"].items():
        lines.append(f"| {POLICY_FR[k]} | {fr(r['per_year'])} | {fr(r['ratio'])} | {r['missed']} sur {r['floods']} | {fr(r['delay_mean_min'])} min | "
                     f"{fr(r['delay_p90_min'])} min | {fr(round(100 * r['within_15_min']))} % | {fr(r['peak_miss_mean_cm'])} cm | {fr(r['peak_miss_max_cm'])} cm |")
    c = out["cost"]
    lines += ["", f"Sur {out['machine']['cpu']}, un thread : une décision du modèle prend {fr(c['fp32_us'])} µs en FP32 et {fr(c['int8_us'])} µs en INT8 "
              f"({c['parameters']} paramètres, {c['macs']} multiplications-additions ; fichier ONNX de {fr(c['fp32_bytes'])} octets en FP32, "
              f"{fr(c['int8_bytes'])} en INT8). INT8 et FP32 prennent la même décision sur {fr(round(100 * c['int8_fp32_agreement'], 2))} % des heures."]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
