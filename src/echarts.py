"""Apache ECharts option generators for reservoir analysis charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# ── Dark theme palette ────────────────────────────────────────────────────────
BG     = '#0a0e1a'
PANEL  = '#111827'
GRID_C = '#1e2d42'
TEXT   = '#e2e8f0'
TEXT2  = '#94a3b8'
BORDER = '#2d3a4f'
BLUE   = '#3b82f6'
GREEN  = '#10b981'
AMBER  = '#f59e0b'
RED    = '#ef4444'
PURPLE = '#8b5cf6'
GOLD   = '#ffd700'
CYAN   = '#06b6d4'

FACIES_COLORS = {0:'#5c3317', 1:'#b8860b', 2:'#daa520', 3:'#7cfc00', 4:'#00e676'}
FACIES_NAMES  = {0:'Shale', 1:'Sandy Shale', 2:'Tight Sand', 3:'Good Sand', 4:'Excellent Sand'}
ZONE_COLORS   = {
    'Overburden':'#6b7280', 'Hugin_Upper':'#f59e0b',
    'Hugin_Middle':'#10b981', 'Hugin_Lower':'#3b82f6', 'Basement':'#9ca3af',
}

# ── Shared axis / tooltip styles ──────────────────────────────────────────────

def _axis(name='', log=False, **kw) -> dict:
    d = dict(
        type='log' if log else 'value',
        name=name,
        nameTextStyle=dict(color=TEXT2, fontSize=11),
        axisLine=dict(lineStyle=dict(color=BORDER)),
        axisTick=dict(lineStyle=dict(color=BORDER)),
        axisLabel=dict(color=TEXT2, fontSize=10),
        splitLine=dict(lineStyle=dict(color=GRID_C, type='dashed')),
        nameGap=35,
    )
    d.update(kw)
    return d

EC_TOOLTIP = dict(
    trigger='item',
    backgroundColor='#1a2332',
    borderColor=BORDER,
    textStyle=dict(color=TEXT, fontSize=11),
)
EC_LEGEND = dict(
    textStyle=dict(color=TEXT2, fontSize=10),
    backgroundColor=PANEL,
    borderColor=BORDER, borderWidth=1, borderRadius=4, padding=8,
)
EC_GRID = dict(left='65', right='30', top='45', bottom='50', containLabel=True)


# ── 1. Variogram ──────────────────────────────────────────────────────────────

def variogram_echart(lags, gamma, h_fit, g_fit, info: dict) -> dict:
    r = float(info.get('range', 0))
    s = float(info.get('sill',  0))
    n = float(info.get('nugget',0))
    m = info.get('model', 'Spherical')
    g_arr  = np.asarray(gamma, float)
    sill_v = round(float(s + n), 9)
    y_max  = max(float(np.max(g_arr)) if len(g_arr) else 0.001, sill_v) * 1.20
    max_lag = float(max(lags)) if len(lags) else 20.0
    return {
        'backgroundColor': BG,
        'tooltip': EC_TOOLTIP,
        'legend': {**EC_LEGEND, 'data': ['Experimental', f'{m} model'], 'top': 4},
        'xAxis': _axis('Lag distance (m)'),
        'yAxis': _axis('Semivariance  γ(h)', min=0, max=round(y_max, 9)),
        'grid': EC_GRID,
        'series': [
            {
                'type': 'scatter',
                'name': 'Experimental',
                'data': [[float(l), float(g)] for l, g in zip(lags, gamma)],
                'itemStyle': {'color': AMBER},
                'symbolSize': 9,
            },
            {
                'type': 'line',
                'name': f'{m} model',
                'data': [[float(h), float(g)] for h, g in zip(h_fit, g_fit)],
                'lineStyle': {'color': BLUE, 'width': 2.5},
                'itemStyle': {'color': BLUE},
                'symbol': 'none',
            },
            # Range vertical line — own series so {xAxis} is never paired with {yAxis}
            {
                'type': 'line', 'data': [], 'silent': True,
                'markLine': {
                    'silent': True,
                    'symbol': ['none', 'none'],
                    'lineStyle': {'color': GREEN, 'width': 1.5, 'type': 'dashed'},
                    'label': {'show': True, 'formatter': f'Range={r:.1f} m',
                              'color': GREEN, 'fontSize': 11, 'position': 'insideEndTop'},
                    'data': [{'xAxis': r}],
                },
            },
            # Sill horizontal line — drawn as a 2-point data series to avoid ECharts
            # integer-truncation bug on {yAxis: small_float} in markLine shortcuts
            {
                'type': 'line',
                'data': [[0.0, sill_v], [max_lag, sill_v]],
                'lineStyle': {'color': RED, 'width': 1.5, 'type': 'dotted'},
                'itemStyle': {'color': RED},
                'symbol': 'none',
                'showSymbol': False,
                'silent': True,
                'endLabel': {
                    'show': True,
                    'formatter': f'Sill={sill_v:.4f}',
                    'color': RED,
                    'fontSize': 10,
                },
            },
        ],
    }


# ── 2. Single layer-metric panel ──────────────────────────────────────────────

def layer_metric_echart(nl: list, vals: list, label: str, opt_n: int,
                        color: str, threshold: float = 0.85,
                        target_pct: float = None) -> dict:
    """Line chart for a single preservation metric vs N layers.

    target_pct: if given (0-1), draw a dashed target line and mark the first N
                that reaches it.
    """
    opt_idx = nl.index(opt_n) if opt_n in nl else None
    mark_pts = []
    if opt_idx is not None:
        mark_pts.append({
            'coord': [opt_n, vals[opt_idx]],
            'symbol': 'pin', 'symbolSize': 18,
            'itemStyle': {'color': GOLD},
            'label': {'show': True, 'formatter': f'N={opt_n}',
                      'color': GOLD, 'fontSize': 10, 'position': 'top'},
        })

    # target preservation marker
    mark_lines = [{
        'yAxis': threshold,
        'lineStyle': {'color': 'rgba(255,255,255,0.22)', 'type': 'dotted'},
        'label': {'formatter': f'{threshold:.0%}', 'color': TEXT2, 'fontSize': 10},
    }]
    if target_pct is not None:
        mark_lines.append({
            'yAxis': target_pct,
            'lineStyle': {'color': RED, 'type': 'dashed', 'width': 1.5},
            'label': {'formatter': f'Target {target_pct:.0%}', 'color': RED, 'fontSize': 10},
        })
        # find first N reaching target
        for i, v in enumerate(vals):
            if v >= target_pct:
                mark_pts.append({
                    'coord': [nl[i], v],
                    'symbol': 'diamond', 'symbolSize': 12,
                    'itemStyle': {'color': RED},
                    'label': {'show': True, 'formatter': f'N={nl[i]}',
                              'color': RED, 'fontSize': 9, 'position': 'right'},
                })
                break

    # area fill colour (hex → rgba)
    def _rgba(hex_c, alpha):
        h = hex_c.lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    return {
        'backgroundColor': BG,
        'tooltip': {**EC_TOOLTIP, 'trigger': 'axis'},
        'xAxis': _axis('N layers'),
        'yAxis': _axis(label, min=0, max=1.05),
        'grid': dict(left='60', right='20', top='30', bottom='45', containLabel=True),
        'series': [{
            'type': 'line',
            'name': label,
            'data': [[int(n), round(float(v), 4)] for n, v in zip(nl, vals)],
            'lineStyle': {'color': color, 'width': 2.5},
            'itemStyle': {'color': color},
            'symbolSize': 4,
            'areaStyle': {'color': {
                'type': 'linear', 'x': 0, 'y': 0, 'x2': 0, 'y2': 1,
                'colorStops': [
                    {'offset': 0, 'color': _rgba(color, 0.22)},
                    {'offset': 1, 'color': 'rgba(0,0,0,0)'},
                ],
            }},
            'markLine': {'silent': True, 'symbol': ['none', 'none'], 'data': mark_lines},
            'markPoint': {'data': mark_pts},
        }],
    }


# ── 3. Cross plot (fixed: no large mode, log-transformed data) ────────────────

def _log10_safe(arr: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(arr, 1e-4, None))


def crossplot_echart(df_all: pd.DataFrame, x_prop: str, y_prop: str,
                     color_by: str = 'FACIES') -> dict:
    """Scatter cross plot.  Uses value axes (log is done in data) for reliability."""
    use_log_y = (y_prop == 'PERM')
    use_log_x = (x_prop == 'PERM')
    series, legend_data = [], []

    def _prep(sub: pd.DataFrame, max_pts: int = 1500):
        xv = sub[x_prop].values.astype(float)
        yv = sub[y_prop].values.astype(float)
        if use_log_x:
            xv = _log10_safe(xv)
        if use_log_y:
            yv = _log10_safe(yv)
        mask = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[mask], yv[mask]
        if len(xv) > max_pts:
            idx = np.random.choice(len(xv), max_pts, replace=False)
            xv, yv = xv[idx], yv[idx]
        return [[round(float(x), 5), round(float(y), 5)] for x, y in zip(xv, yv)]

    if color_by == 'FACIES' and 'FACIES' in df_all.columns:
        for g in sorted(df_all['FACIES'].unique()):
            sub  = df_all[df_all['FACIES'] == g]
            name = FACIES_NAMES.get(int(g), str(g))
            clr  = FACIES_COLORS.get(int(g), '#888888')
            series.append({
                'type': 'scatter', 'name': name,
                'data': _prep(sub),
                'itemStyle': {'color': clr, 'opacity': 0.72},
                'symbolSize': 4,
            })
            legend_data.append(name)
    else:
        zones = sorted(df_all['ZONE'].unique()) if 'ZONE' in df_all.columns else []
        for z in zones:
            sub  = df_all[df_all['ZONE'] == z]
            name = z.replace('_', ' ')
            clr  = ZONE_COLORS.get(z, '#888888')
            series.append({
                'type': 'scatter', 'name': name,
                'data': _prep(sub),
                'itemStyle': {'color': clr, 'opacity': 0.72},
                'symbolSize': 4,
            })
            legend_data.append(name)

    x_label = ('log₁₀ ' if use_log_x else '') + x_prop
    y_label = ('log₁₀ ' if use_log_y else '') + y_prop

    return {
        'backgroundColor': BG,
        'tooltip': {**EC_TOOLTIP, 'trigger': 'item',
                    'formatter': f'{x_label}: {{c[0]}}<br/>{y_label}: {{c[1]}}'},
        'legend': {**EC_LEGEND, 'data': legend_data, 'type': 'scroll', 'top': 4},
        'xAxis': _axis(x_label),
        'yAxis': _axis(y_label),
        'grid': EC_GRID,
        'series': series,
        'dataZoom': [
            {'type': 'inside', 'xAxisIndex': 0, 'filterMode': 'none'},
            {'type': 'inside', 'yAxisIndex': 0, 'filterMode': 'none'},
            {'type': 'slider', 'xAxisIndex': 0, 'height': 16, 'bottom': 4,
             'textStyle': {'color': TEXT2},
             'fillerColor': 'rgba(59,130,246,0.15)', 'borderColor': BORDER},
        ],
    }


# ── 4. Heterogeneity vs Model Layers (GeoConvention 2010) ────────────────────

def heterogeneity_echart(res: pd.DataFrame, zone_name: str = '',
                         target_pct: float = None) -> dict:
    """Heterogeneity preserved (%) vs N layers — Fig.1 of GeoConv 2010 paper.

    Curve goes from low heterogeneity (few layers) to 100 % (fine grid).
    Elbow = optimal N (gold pin).  target_pct adds a horizontal target line.
    """
    if res is None or res.empty:
        return {'backgroundColor': BG, 'series': []}

    nl   = res['n_layers'].values.tolist()
    hp   = res['heterogeneity_pct'].values.tolist()
    opt_row = res[res['inflection']]
    opt_n   = int(opt_row['n_layers'].iloc[0])   if len(opt_row) else nl[-1]
    opt_h   = float(opt_row['heterogeneity_pct'].iloc[0]) if len(opt_row) else hp[-1]

    mark_lines = []
    mark_pts   = [{
        'coord': [opt_n, opt_h],
        'symbol': 'pin', 'symbolSize': 20,
        'itemStyle': {'color': GOLD},
        'label': {'show': True, 'formatter': f'Elbow N={opt_n}',
                  'color': GOLD, 'fontSize': 10, 'position': 'top'},
    }]

    if target_pct is not None:
        mark_lines.append({
            'yAxis': target_pct,
            'lineStyle': {'color': RED, 'type': 'dashed', 'width': 1.5},
            'label': {'formatter': f'Target {target_pct:.0f}%', 'color': RED, 'fontSize': 10},
        })
        for n_v, h_v in zip(nl, hp):
            if h_v >= target_pct:
                mark_pts.append({
                    'coord': [n_v, h_v],
                    'symbol': 'diamond', 'symbolSize': 14,
                    'itemStyle': {'color': RED},
                    'label': {'show': True, 'formatter': f'N={n_v}',
                              'color': RED, 'fontSize': 10, 'position': 'right'},
                })
                break

    return {
        'backgroundColor': BG,
        'tooltip': {**EC_TOOLTIP, 'trigger': 'axis',
                    'formatter': 'N layers: {b}<br/>Heterogeneity preserved: {c}%'},
        'xAxis': _axis('Number of Layers (N)', min=1),
        'yAxis': _axis('Heterogeneity Preserved (%)', min=0, max=100),
        'grid': dict(left='65', right='30', top='40', bottom='50', containLabel=True),
        'series': [{
            'type': 'line',
            'name': 'Heterogeneity',
            'data': [[int(n), round(float(h), 3)] for n, h in zip(nl, hp)],
            'lineStyle': {'color': CYAN, 'width': 2.5},
            'itemStyle': {'color': CYAN},
            'symbolSize': 4,
            'areaStyle': {'color': {
                'type': 'linear', 'x': 0, 'y': 0, 'x2': 0, 'y2': 1,
                'colorStops': [
                    {'offset': 0, 'color': 'rgba(6,182,212,0.22)'},
                    {'offset': 1, 'color': 'rgba(6,182,212,0)'},
                ],
            }},
            'markLine': {'silent': True, 'symbol': ['none', 'none'], 'data': mark_lines},
            'markPoint': {'data': mark_pts},
        }],
    }


# ── 5. Grid search heatmap ────────────────────────────────────────────────────

def grid_search_heatmap(all_results: dict) -> dict:
    """ECharts heatmap — Y = zones, X = N layers, colour = combined_score."""
    zones = list(all_results.keys())
    if not zones:
        return {'backgroundColor': BG, 'series': []}

    max_n  = max(int(df['n_layers'].max()) for df in all_results.values() if not df.empty)
    n_cats = [str(i) for i in range(1, max_n + 1)]

    heat_data, star_data = [], []
    score_min, score_max = 1.0, 0.0

    for zi, zname in enumerate(zones):
        df = all_results[zname]
        if df.empty:
            continue
        opt_n = int(df.loc[df['optimal'], 'n_layers'].iloc[0]) if df['optimal'].any() else 1
        for _, row in df.iterrows():
            n = int(row['n_layers'])
            s = float(row['combined_score'])
            score_min = min(score_min, s)
            score_max = max(score_max, s)
            heat_data.append([n - 1, zi, round(s, 4)])
            if n == opt_n:
                star_data.append([n - 1, zi, f'N={n}'])

    return {
        'backgroundColor': BG,
        'tooltip': {
            'trigger': 'item',
            'backgroundColor': '#1a2332',
            'borderColor': BORDER,
            'textStyle': {'color': TEXT, 'fontSize': 11},
        },
        'grid': {'left': '10', 'right': '10', 'top': '30', 'bottom': '65', 'containLabel': True},
        'xAxis': {
            'type': 'category',
            'data': n_cats,
            'name': 'N Layers',
            'nameLocation': 'middle',
            'nameGap': 32,
            'nameTextStyle': {'color': TEXT2, 'fontSize': 11},
            'axisLabel': {'color': TEXT2, 'fontSize': 9,
                          'interval': max(0, len(n_cats) // 20 - 1)},
            'axisLine': {'lineStyle': {'color': BORDER}},
            'splitLine': {'show': False},
        },
        'yAxis': {
            'type': 'category',
            'data': [z.replace('_', ' ') for z in zones],
            'axisLabel': {'color': TEXT2, 'fontSize': 10},
            'axisLine': {'lineStyle': {'color': BORDER}},
        },
        'visualMap': {
            'min': round(score_min, 3),
            'max': round(score_max, 3),
            'calculable': True,
            'orient': 'horizontal',
            'left': 'center',
            'bottom': 5,
            'itemWidth': 14,
            'itemHeight': 120,
            'textStyle': {'color': TEXT2, 'fontSize': 9},
            'inRange': {'color': ['#0a1628', '#1e3a5f', '#3b82f6', '#10b981', '#f59e0b', '#ffd700']},
        },
        'series': [
            {
                'type': 'heatmap',
                'name': 'Score',
                'data': heat_data,
                'label': {'show': False},
                'emphasis': {'itemStyle': {'shadowBlur': 8, 'shadowColor': GOLD}},
            },
            {
                'type': 'scatter',
                'name': 'Optimal N',
                'data': [{'value': [d[0], d[1]],
                           'label': {'show': True, 'formatter': d[2],
                                     'color': '#111', 'fontSize': 8,
                                     'fontWeight': 'bold', 'position': 'inside'}}
                          for d in star_data],
                'symbol': 'rect',
                'symbolSize': [18, 18],
                'itemStyle': {'color': GOLD, 'opacity': 0.92,
                              'borderColor': '#111', 'borderWidth': 1},
                'zlevel': 2,
            },
        ],
    }


# ── 6. Zone thickness per well (replaces NTG variogram) ──────────────────────

def zone_thickness_echart(well_names: list, thicknesses: list, zone_name: str = '') -> dict:
    """Horizontal bar chart — zone thickness per well for selected zone."""
    pairs = sorted(zip(thicknesses, well_names), key=lambda x: x[0])
    y_cats = [w for _, w in pairs]
    x_vals = [round(float(t), 1) for t, _ in pairs]
    avg_t  = float(np.mean(thicknesses)) if thicknesses else 0.0
    return {
        'backgroundColor': BG,
        'tooltip': {**EC_TOOLTIP,
                    'formatter': '{b}: {c} m'},
        'grid': dict(left='80', right='60', top='30', bottom='30', containLabel=True),
        'xAxis': _axis(f'Thickness (m)', min=0),
        'yAxis': {
            'type': 'category',
            'data': y_cats,
            'axisLabel': {'color': TEXT2, 'fontSize': 10},
            'axisLine': {'lineStyle': {'color': BORDER}},
            'axisTick': {'lineStyle': {'color': BORDER}},
        },
        'series': [
            {
                'type': 'bar',
                'name': 'Thickness',
                'data': x_vals,
                'barMaxWidth': 28,
                'itemStyle': {'color': GREEN, 'borderRadius': [0, 3, 3, 0]},
                'label': {'show': True, 'position': 'right',
                          'formatter': '{c} m',
                          'color': TEXT, 'fontSize': 9},
            },
            {
                'type': 'line',
                'name': 'Avg',
                'data': [avg_t] * len(y_cats),
                'lineStyle': {'color': AMBER, 'type': 'dashed', 'width': 1.5},
                'itemStyle': {'color': AMBER},
                'symbol': 'none',
                'markLine': {
                    'silent': True,
                    'symbol': ['none', 'none'],
                    'data': [{'xAxis': avg_t,
                              'lineStyle': {'color': AMBER, 'type': 'dashed', 'width': 1.5},
                              'label': {'formatter': f'Avg {avg_t:.1f} m',
                                        'color': AMBER, 'fontSize': 10,
                                        'position': 'insideEndTop'}}],
                },
            },
        ],
    }


# ── 7. Stats table rows ───────────────────────────────────────────────────────

def stats_rows(results: pd.DataFrame) -> List[dict]:
    opt = results[results['optimal']].iloc[0] if results['optimal'].any() else results.iloc[-1]
    def badge(v, t): return '✓ Good' if v >= t else '⚠ Low'
    rows = [
        {'metric': 'Recommended N layers',  'value': str(int(opt['n_layers'])),              'status': '★ Optimal'},
        {'metric': 'VP threshold N (≥85%)', 'value': str(int(opt['vp_threshold_n'])),         'status': ''},
        {'metric': 'Variance Preservation', 'value': f"{opt['variance_preservation']:.1%}",   'status': badge(opt['variance_preservation'], 0.85)},
        {'metric': 'Lorenz Preservation',   'value': f"{opt['lorenz_preservation']:.1%}",     'status': badge(opt['lorenz_preservation'],   0.85)},
        {'metric': 'DP Preservation',       'value': f"{opt['dp_preservation']:.1%}",         'status': badge(opt['dp_preservation'],        0.80)},
        {'metric': 'Facies Coverage',       'value': f"{opt['facies_coverage']:.1%}",         'status': badge(opt['facies_coverage'],        1.00)},
        {'metric': 'Combined Score',        'value': f"{opt['combined_score']:.3f}",          'status': ''},
        {'metric': 'Lorenz Coeff (orig.)',  'value': f"{opt['lorenz_coeff_orig']:.3f}",       'status': ''},
        {'metric': 'DP Coeff (orig.)',      'value': f"{opt['dp_coeff_orig']:.3f}",           'status': ''},
    ]
    return rows
