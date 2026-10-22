"""Download the 5-minute water level series of six French gauging stations, one request per station-year.

    python -m src.fetch        # writes data/<station>.npz (skips what is already there)

Source: HydroPortail (SC Vigicrues / Eaufrance), "most valid" instantaneous heights. This is the endpoint the
site itself calls to draw its charts and build its CSV export; the same data can be exported by hand from
https://www.hydro.eaufrance.fr/stationhydro/<code>/series. Requests are sequential, with a pause, and cached.
"""
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
YEARS = range(2019, 2026)
# Station codes from the Hub'Eau station register. The Garonne is a large river; the five others are
# Mediterranean rivers known for flash floods.
STATIONS = {
    "O200004001": "La Garonne à Toulouse",
    "V714401001": "Le Gard à Anduze",
    "Y345401001": "Le Vidourle à Sommières",
    "V505401001": "L'Ardèche à Vallon-Pont-d'Arc",
    "Y123201002": "L'Aude à Carcassonne",
    "Y210002001": "L'Hérault à Laroque",
}
URL = "https://www.hydro.eaufrance.fr/stationhydro/ajax/{code}/series?"


def fetch_year(code: str, year: int) -> tuple[np.ndarray, np.ndarray]:
    """(unix seconds, height in metres) of one calendar year."""
    q = {"hydro_series[startAt]": f"01/01/{year}", "hydro_series[endAt]": f"31/12/{year}",
         "hydro_series[variableType]": "simple_and_interpolated_and_hourly_variable",
         "hydro_series[simpleAndInterpolatedAndHourlyVariable]": "H", "hydro_series[statusData]": "most_valid"}
    req = urllib.request.Request(URL.format(code=code) + urllib.parse.urlencode(q),
                                 headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                rows = json.load(r)["series"]["data"]
            break
        except (OSError, ValueError, KeyError):
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))
    t = np.array([datetime.fromisoformat(p["t"].replace("Z", "+00:00")).timestamp() for p in rows], dtype=np.int64)
    h = np.array([p["v"] for p in rows], dtype=np.float32) / 1000  # mm -> m
    return t, h


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for code, name in STATIONS.items():
        out = DATA / f"{code}.npz"
        if out.exists():
            continue
        ts, hs = [], []
        for year in YEARS:
            t, h = fetch_year(code, year)
            ts.append(t)
            hs.append(h)
            print(f"{name} {year}: {len(t)} points")
            time.sleep(2)
        np.savez_compressed(out, t=np.concatenate(ts), h=np.concatenate(hs))


def load(code: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(DATA / f"{code}.npz") as f:
        return f["t"], f["h"]


if __name__ == "__main__":
    main()
