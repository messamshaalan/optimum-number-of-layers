"""Synthetic Volve-like reservoir dataset generator."""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from typing import Dict, Tuple

WELL_NAMES = ['F-1H', 'F-3', 'F-4', 'F-5', 'F-6']

WELL_POSITIONS: Dict[str, Tuple[float, float]] = {
    'F-1H': (437000, 6476000),
    'F-3':  (437800, 6476500),
    'F-4':  (438200, 6475800),
    'F-5':  (436500, 6476800),
    'F-6':  (438500, 6476300),
}

# (top_md, base_md) in metres
ZONES = {
    'Overburden':    (2500, 2650),
    'Hugin_Upper':   (2650, 2710),
    'Hugin_Middle':  (2710, 2755),
    'Hugin_Lower':   (2755, 2820),
    'Basement':      (2820, 2900),
}

RESERVOIR_ZONES = ['Hugin_Upper', 'Hugin_Middle', 'Hugin_Lower']
FWL = 2830  # Free Water Level (m TVDSS)

FACIES_NAMES = {
    0: 'Shale',
    1: 'Sandy Shale',
    2: 'Tight Sand',
    3: 'Good Sand',
    4: 'Excellent Sand',
}

# Transition matrices per environment
_TM_OVERBURDEN = np.array([
    [0.92, 0.06, 0.02, 0.00, 0.00],
    [0.30, 0.55, 0.13, 0.02, 0.00],
    [0.20, 0.20, 0.52, 0.07, 0.01],
    [0.05, 0.10, 0.32, 0.50, 0.03],
    [0.02, 0.04, 0.18, 0.42, 0.34],
])
_TM_RESERVOIR = np.array([
    [0.55, 0.18, 0.17, 0.08, 0.02],
    [0.12, 0.38, 0.28, 0.18, 0.04],
    [0.08, 0.13, 0.40, 0.30, 0.09],
    [0.02, 0.06, 0.18, 0.52, 0.22],
    [0.01, 0.02, 0.10, 0.34, 0.53],
])
_IP_OVERBURDEN = [0.82, 0.11, 0.05, 0.02, 0.00]
_IP_RESERVOIR  = [0.18, 0.14, 0.24, 0.27, 0.17]


def _markov_facies(n: int, transition: np.ndarray, initial: list, rng) -> np.ndarray:
    facies = np.empty(n, dtype=np.int8)
    facies[0] = rng.choice(5, p=initial)
    for i in range(1, n):
        facies[i] = rng.choice(5, p=transition[facies[i - 1]])
    return facies


def _generate_logs(depth: np.ndarray, zones: np.ndarray, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = len(depth)

    # --- Facies column (Markov chain, per zone segment) ---
    facies = np.empty(n, dtype=np.int8)
    i = 0
    while i < n:
        z = zones[i]
        is_res = z in RESERVOIR_ZONES
        tm = _TM_RESERVOIR if is_res else _TM_OVERBURDEN
        ip = _IP_RESERVOIR  if is_res else _IP_OVERBURDEN
        # find end of this zone segment
        j = i + 1
        while j < n and zones[j] == z:
            j += 1
        seg_len = j - i
        facies[i:j] = _markov_facies(seg_len, tm, ip, rng)
        i = j

    # --- GR (API) ---
    gr_mean = np.array([112, 86, 65, 40, 24])[facies]
    gr_std  = np.array([14,  12, 10,  8,  6])[facies]
    gr = gr_mean + rng.randn(n) * gr_std
    gr = gaussian_filter1d(gr, sigma=2.5)
    gr = np.clip(gr, 10, 160)

    # --- PHIE ---
    phi_mean = np.array([0.040, 0.080, 0.125, 0.220, 0.280])[facies]
    phi_std  = np.array([0.012, 0.018, 0.022, 0.028, 0.022])[facies]
    phie = phi_mean + rng.randn(n) * phi_std
    phie = gaussian_filter1d(phie, sigma=1.8)
    phie = np.clip(phie, 0.005, 0.40)

    # --- PERM (log-normal, correlated with PHIE) ---
    lk_mean = np.array([-2.0, -0.4, 0.9, 2.6, 3.3])[facies]
    lk_std  = np.array([ 0.5,  0.6, 0.7, 0.6, 0.5])[facies]
    log_k = lk_mean + rng.randn(n) * lk_std + 3.0 * (phie - np.mean(phie))
    log_k = gaussian_filter1d(log_k, sigma=1.8)
    perm = np.clip(10 ** log_k, 0.001, 15000)

    # --- NTG (GR cutoff = 65 API) ---
    ntg = gaussian_filter1d((gr < 65).astype(float), sigma=1.5)
    ntg = np.clip(ntg, 0.0, 1.0)

    # --- Sw (Archie-like above FWL) ---
    sw = np.ones(n)
    for idx in range(n):
        dist = FWL - depth[idx]
        if dist > 0 and facies[idx] >= 2:
            sw_min = max(0.10, 0.11 + 0.015 * rng.randn())
            sw[idx] = sw_min + (0.55 - sw_min) * np.exp(-dist / 55.0)
        elif dist <= 0:
            sw[idx] = 1.0
        else:
            sw[idx] = 0.95 + rng.randn() * 0.03
    sw = gaussian_filter1d(sw, sigma=2.0)
    sw = np.clip(sw, 0.08, 1.0)

    # --- RHOB (g/cc) ---
    rhob = 2.71 - 0.95 * phie + rng.randn(n) * 0.025
    rhob = gaussian_filter1d(rhob, sigma=1.2)
    rhob = np.clip(rhob, 1.95, 2.88)

    # --- NPHI ---
    nphi = phie + 0.018 + rng.randn(n) * 0.018
    nphi = gaussian_filter1d(nphi, sigma=1.2)
    nphi = np.clip(nphi, 0.0, 0.50)

    return pd.DataFrame({
        'GR':     gr,
        'PHIE':   phie,
        'PERM':   perm,
        'SW':     sw,
        'NTG':    ntg,
        'RHOB':   rhob,
        'NPHI':   nphi,
        'FACIES': facies.astype(int),
    })


def generate_synthetic_wells() -> Dict[str, pd.DataFrame]:
    """Return dict of well_name → DataFrame with DEPTH, ZONE, X, Y, TVDSS, and log columns."""
    wells = {}
    dip_offsets = [-12, -4, 8, 16, -8]  # structural variation between wells (m)

    for wi, well_name in enumerate(WELL_NAMES):
        x0, y0 = WELL_POSITIONS[well_name]
        depth = np.arange(2500.0, 2900.5, 0.5)
        n = len(depth)

        # Zone assignment
        zone_arr = np.empty(n, dtype=object)
        for d_idx, d in enumerate(depth):
            zone_arr[d_idx] = 'Basement'
            for zname, (ztop, zbase) in ZONES.items():
                if ztop <= d < zbase:
                    zone_arr[d_idx] = zname
                    break

        logs = _generate_logs(depth, zone_arr, seed=wi * 137 + 31)

        # Slight deviation for directional wells
        dev_x = [0, 120, 200, 40, 180][wi]
        dev_y = [0,  60, 100, 20,  90][wi]
        xs = x0 + np.linspace(0, dev_x, n)
        ys = y0 + np.linspace(0, dev_y, n)
        tvd = depth + dip_offsets[wi]

        df = pd.DataFrame({'DEPTH': depth, 'ZONE': zone_arr, 'X': xs, 'Y': ys, 'TVDSS': tvd})
        df = pd.concat([df.reset_index(drop=True), logs.reset_index(drop=True)], axis=1)
        wells[well_name] = df

    return wells


def generate_formation_tops(wells: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return DataFrame: WELL, ZONE, TOP, BASE, THICKNESS, X, Y."""
    dip_offsets = {'F-1H': -12, 'F-3': -4, 'F-4': 8, 'F-5': 16, 'F-6': -8}
    rows = []
    for well_name in WELL_NAMES:
        x0, y0 = WELL_POSITIONS[well_name]
        off = dip_offsets[well_name]
        for zname, (ztop, zbase) in ZONES.items():
            rows.append({
                'WELL':      well_name,
                'ZONE':      zname,
                'TOP':       ztop + off,
                'BASE':      zbase + off,
                'THICKNESS': zbase - ztop,
                'X': x0, 'Y': y0,
            })
    return pd.DataFrame(rows)
