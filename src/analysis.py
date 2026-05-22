"""Core reservoir analysis: variogram, upscaling, layer optimisation."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Dict


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def lorenz_coefficient(phi: np.ndarray, k: np.ndarray) -> float:
    """Lorenz coefficient (0 = homogeneous, 1 = extremely heterogeneous)."""
    phi = np.asarray(phi, float)
    k   = np.asarray(k,   float)
    mask = (phi > 0) & (k > 0)
    if mask.sum() < 2:
        return 0.0
    phi, k = phi[mask], k[mask]
    order = np.argsort(k / phi)[::-1]          # sort by flow capacity ratio
    cum_phi = np.cumsum(phi[order]) / phi.sum()
    cum_k   = np.cumsum(k[order])   / k.sum()
    lc = float(2.0 * np.trapezoid(cum_k, cum_phi) - 1.0)
    return max(0.0, min(1.0, lc))


def dykstra_parsons_v(k: np.ndarray) -> float:
    """Dykstra-Parsons coefficient from permeability array."""
    k = np.asarray(k, float)
    k = k[k > 0]
    if len(k) < 3:
        return 0.0
    lk = np.log(k)
    p50 = np.percentile(lk, 50)
    p84 = np.percentile(lk, 84.1)
    dp = 1.0 - np.exp(p50 - p84)
    return float(np.clip(dp, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Variogram analysis
# ---------------------------------------------------------------------------

def compute_variogram(
    z: np.ndarray, dz: float, n_lags: int = 25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-D variogram along depth direction (Matheron estimator).

    Returns lags (m), gamma (semivariance), n_pairs.
    """
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    n = len(z)
    lags   = np.arange(1, n_lags + 1, dtype=float) * dz
    gamma  = np.zeros(n_lags)
    npairs = np.zeros(n_lags, dtype=int)
    for ki in range(1, n_lags + 1):
        diff = z[ki:] - z[:-ki]
        gamma[ki - 1]  = 0.5 * np.mean(diff ** 2)
        npairs[ki - 1] = len(diff)
    return lags, gamma, npairs


def _spherical(h, nugget, sill, a):
    ratio = h / a
    return np.where(h <= a,
                    nugget + sill * (1.5 * ratio - 0.5 * ratio ** 3),
                    nugget + sill)


def _exponential(h, nugget, sill, a):
    return nugget + sill * (1.0 - np.exp(-3.0 * h / a))


def _gaussian(h, nugget, sill, a):
    return nugget + sill * (1.0 - np.exp(-3.0 * (h / a) ** 2))


_MODEL_FUNCS = {
    'Spherical':    _spherical,
    'Exponential':  _exponential,
    'Gaussian':     _gaussian,
}


def fit_variogram(
    lags: np.ndarray, gamma: np.ndarray, model_type: str = 'Spherical'
) -> Dict:
    """Fit theoretical variogram model, return dict with fitted parameters."""
    func = _MODEL_FUNCS.get(model_type, _spherical)
    sill0   = float(np.percentile(gamma, 90))
    nugget0 = float(gamma[0] * 0.1)
    range0  = float(lags[np.argmax(gamma >= sill0 * 0.90)] if any(gamma >= sill0 * 0.90) else lags[-1] * 0.5)

    try:
        popt, _ = curve_fit(
            func, lags, gamma,
            p0=[nugget0, sill0, range0],
            bounds=([0, 0, lags[0]], [sill0 * 0.5, sill0 * 2, lags[-1] * 3]),
            maxfev=5000,
        )
        nugget, sill, range_ = popt
    except Exception:
        nugget, sill, range_ = nugget0, sill0, range0

    h_fit = np.linspace(0, lags[-1] * 1.1, 200)
    g_fit = func(h_fit, nugget, sill, range_)

    return {
        'model':   model_type,
        'nugget':  float(nugget),
        'sill':    float(sill),
        'range':   float(range_),
        'h_fit':   h_fit,
        'g_fit':   g_fit,
    }


# ---------------------------------------------------------------------------
# Upscaling
# ---------------------------------------------------------------------------

def upscale_zone(df: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    """Upscale well logs in df to n_layers equal-thickness layers.

    Returns DataFrame with one row per upscaled layer:
    LAYER, DEPTH_TOP, DEPTH_BASE, DEPTH_MID, GR, PHIE, PERM, SW, NTG, FACIES
    """
    n = len(df)
    if n == 0 or n_layers <= 0:
        return pd.DataFrame()

    n_layers = min(n_layers, n)
    boundaries = np.round(np.linspace(0, n, n_layers + 1)).astype(int)
    depth = df['DEPTH'].values

    rows = []
    for li in range(n_layers):
        s, e = boundaries[li], boundaries[li + 1]
        if e <= s:
            continue
        seg = df.iloc[s:e]
        phie  = seg['PHIE'].values
        perm  = seg['PERM'].values
        facies = seg['FACIES'].values

        rows.append({
            'LAYER':      li + 1,
            'DEPTH_TOP':  depth[s],
            'DEPTH_BASE': depth[min(e, n - 1)],
            'DEPTH_MID':  0.5 * (depth[s] + depth[min(e, n - 1)]),
            'GR':   float(np.mean(seg['GR'])),
            'PHIE': float(np.mean(phie)),
            'PERM': float(np.exp(np.mean(np.log(np.clip(perm, 1e-6, None))))),  # geometric mean
            'SW':   float(np.mean(seg['SW'])),
            'NTG':  float(np.mean(seg['NTG'])),
            'FACIES': int(np.round(np.mean(facies))),
            'FACIES_MODE': int(np.bincount(facies).argmax()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer optimisation
# ---------------------------------------------------------------------------

def optimize_layers(df: pd.DataFrame, max_layers: int = 40) -> pd.DataFrame:
    """Compute preservation metrics for N = 1..max_layers.

    Returns DataFrame with columns:
    n_layers, variance_preservation, lorenz_preservation,
    facies_coverage, ntg_preservation, combined_score, marginal_gain, optimal
    """
    n = len(df)
    if n < 2:
        return pd.DataFrame()

    max_layers = min(max_layers, n)

    phie   = df['PHIE'].values
    perm   = df['PERM'].values
    facies = df['FACIES'].values
    ntg    = df['NTG'].values

    var_orig = float(np.var(phie)) + 1e-12
    lc_orig  = lorenz_coefficient(phie, perm)
    dp_orig  = dykstra_parsons_v(perm)
    n_facies = len(np.unique(facies))
    ntg_mean = float(np.mean(ntg))

    records = []
    for nl in range(1, max_layers + 1):
        up = upscale_zone(df, nl)
        if up.empty:
            continue

        up_phi   = up['PHIE'].values
        up_perm  = up['PERM'].values
        up_fac   = up['FACIES_MODE'].values
        up_ntg   = up['NTG'].values

        # Variance preservation
        vp = min(1.0, float(np.var(up_phi)) / var_orig)

        # Lorenz preservation
        lc_up = lorenz_coefficient(up_phi, up_perm)
        lcp = min(1.0, lc_up / (lc_orig + 1e-9)) if lc_orig > 1e-9 else 1.0

        # Dykstra-Parsons preservation
        dp_up  = dykstra_parsons_v(up_perm)
        dpp = min(1.0, dp_up / (dp_orig + 1e-9)) if dp_orig > 1e-9 else 1.0

        # Facies coverage
        fc = len(np.unique(up_fac)) / max(n_facies, 1)

        # NTG preservation
        ntg_err = abs(float(np.mean(up_ntg)) - ntg_mean) / (ntg_mean + 1e-9)
        ntgp = max(0.0, 1.0 - ntg_err)

        # Combined score (weighted)
        score = 0.30 * vp + 0.25 * lcp + 0.20 * dpp + 0.15 * fc + 0.10 * ntgp

        records.append({
            'n_layers':              nl,
            'variance_preservation': round(vp,  4),
            'lorenz_preservation':   round(lcp, 4),
            'dp_preservation':       round(dpp, 4),
            'facies_coverage':       round(fc,  4),
            'ntg_preservation':      round(ntgp, 4),
            'combined_score':        round(score, 4),
        })

    if not records:
        return pd.DataFrame()

    res = pd.DataFrame(records)

    # Marginal gain
    res['marginal_gain'] = res['combined_score'].diff().fillna(res['combined_score'].iloc[0])

    # Elbow: optimal N is the last layer where marginal gain > 2 % of max gain
    max_gain = res['marginal_gain'].max()
    threshold = max(0.01, max_gain * 0.06)
    qualifying = res[res['marginal_gain'] >= threshold]
    opt_n = int(qualifying['n_layers'].max()) if len(qualifying) else 1

    res['optimal'] = res['n_layers'] == opt_n

    # Variance-threshold recommendation (≥85 % preservation)
    vp_thresh = res[res['variance_preservation'] >= 0.85]
    vp_n = int(vp_thresh['n_layers'].min()) if len(vp_thresh) else opt_n

    res['recommended_n']     = opt_n
    res['vp_threshold_n']    = vp_n
    res['lorenz_coeff_orig'] = round(lc_orig, 4)
    res['dp_coeff_orig']     = round(dp_orig, 4)

    return res


# ---------------------------------------------------------------------------
# Variogram-based layer count recommendation
# ---------------------------------------------------------------------------

def compute_model_curve(
    lags: np.ndarray, model: str, nugget: float, sill: float, range_: float
) -> np.ndarray:
    """Evaluate a named variogram model at the given lags."""
    func = _MODEL_FUNCS.get(model, _spherical)
    h = np.asarray(lags, float)
    return func(h, float(nugget), float(sill), float(range_))


def recommend_from_variogram(zone_thickness: float, vertical_range: float) -> int:
    """Minimum layer count so that layer thickness ≤ vertical range."""
    if vertical_range <= 0:
        return 10
    n = int(np.ceil(zone_thickness / vertical_range))
    return max(1, n)


# ---------------------------------------------------------------------------
# HCPV proxy
# ---------------------------------------------------------------------------

def compute_hcpv_proxy(df: pd.DataFrame) -> np.ndarray:
    """Hydrocarbon pore volume proxy per sample: PHIE × NTG × (1-Sw) × dz."""
    dz   = np.abs(df['DEPTH'].diff().fillna(df['DEPTH'].diff().median())).values
    phie = np.clip(df['PHIE'].values, 0.0, 1.0)
    ntg  = np.clip(df['NTG'].values, 0.0, 1.0) if 'NTG' in df.columns else np.ones(len(df))
    sw   = np.clip(df['SW'].values,  0.0, 1.0) if 'SW'  in df.columns else np.zeros(len(df))
    return phie * ntg * np.clip(1.0 - sw, 0.0, 1.0) * dz


# ---------------------------------------------------------------------------
# GeoConvention 2010 heterogeneity coarsening algorithm
# "A Workflow for Optimal Simulation Grid Design"
# ---------------------------------------------------------------------------

def _find_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """Return the x-value at the elbow of the curve (max perpendicular distance)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return int(x[0])
    rng_x = x.max() - x.min() + 1e-12
    rng_y = y.max() - y.min() + 1e-12
    xn = (x - x.min()) / rng_x
    yn = (y - y.min()) / rng_y
    d  = np.array([xn[-1] - xn[0], yn[-1] - yn[0]])
    d /= np.linalg.norm(d) + 1e-12
    perp = np.array([-d[1], d[0]])
    pts  = np.column_stack([xn - xn[0], yn - yn[0]])
    dists = np.abs(pts @ perp)
    return int(x[np.argmax(dists)])


def heterogeneity_coarsening(
    df: pd.DataFrame,
    perm_col: str = 'PERM',
    phi_col:  str = 'PHIE',
) -> pd.DataFrame:
    """Greedy merge algorithm from GeoConvention 2010.

    Local velocity  P  = K / phi  (Equation 1 — Buckley-Leverett speed proxy)
    Total heterogeneity H = Σ n_k (P_k - P̄)²          (Equation 2)
    Variation change    W_k = (n_k n_{k+1}/(n_k+n_{k+1})) (P_k-P_{k+1})²  (Eq. 4)
    At each step the pair with minimum W_k is merged.

    Returns DataFrame:
        n_layers, heterogeneity_pct, h_abs, inflection (bool), h_lost_pct
    """
    df = df.reset_index(drop=True)
    n  = len(df)
    if n < 2:
        return pd.DataFrame()

    dz   = np.abs(df['DEPTH'].diff().fillna(df['DEPTH'].diff().median())).values
    phi  = np.clip(df[phi_col].values, 1e-4, None)
    perm = np.clip(df[perm_col].values, 1e-4, None)

    P_cur = perm / phi                  # local velocity (Eq. 1)
    n_cur = dz.copy()                   # bulk-volume proxy (thickness × unit area)

    def _h_total(nv, Pv):
        p_bar = np.dot(nv, Pv) / (nv.sum() + 1e-15)
        return float(np.dot(nv, (Pv - p_bar) ** 2))

    H0 = _h_total(n_cur, P_cur)
    if H0 < 1e-15:
        H0 = 1e-15

    records = [{'n_layers': n, 'h_abs': H0, 'heterogeneity_pct': 100.0, 'h_lost_pct': 0.0}]

    H_cur = H0
    for _ in range(n - 1):
        m = len(P_cur)
        if m < 2:
            break
        # Variation change for each adjacent pair (Eq. 4)
        nk  = n_cur[:-1]
        nk1 = n_cur[1:]
        Wk  = (nk * nk1 / (nk + nk1 + 1e-15)) * (P_cur[:-1] - P_cur[1:]) ** 2
        k_star = int(np.argmin(Wk))
        W_min  = float(Wk[k_star])

        # Merge layers k* and k*+1 (volume-weighted mean, Eq. 6)
        n_new = n_cur[k_star] + n_cur[k_star + 1]
        P_new = (n_cur[k_star] * P_cur[k_star] + n_cur[k_star + 1] * P_cur[k_star + 1]) / n_new
        P_cur = np.concatenate([P_cur[:k_star], [P_new], P_cur[k_star + 2:]])
        n_cur = np.concatenate([n_cur[:k_star], [n_new], n_cur[k_star + 2:]])

        H_cur = max(0.0, H_cur - W_min)
        h_pct = 100.0 * H_cur / H0
        records.append({
            'n_layers':         int(len(P_cur)),
            'h_abs':            H_cur,
            'heterogeneity_pct': round(h_pct, 3),
            'h_lost_pct':       round(100.0 - h_pct, 3),
        })

    res = pd.DataFrame(records).sort_values('n_layers').reset_index(drop=True)

    # Elbow = optimal N (max perpendicular distance from diagonal)
    opt_n = _find_elbow(res['n_layers'].values, res['heterogeneity_pct'].values)
    res['inflection'] = res['n_layers'] == opt_n
    return res
