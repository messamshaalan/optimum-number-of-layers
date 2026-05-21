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
    return {
        'backgroundColor': BG,
        'tooltip': EC_TOOLTIP,
        'legend': {**EC_LEGEND, 'data': ['Experimental', f'{m} model'], 'top': 4},
        'xAxis': _axis('Lag distance (m)'),
        'yAxis': _axis('Semivariance  γ(h)'),
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
                'markLine': {
                    'silent': True,
                    'symbol': ['none', 'none'],
                    'lineStyle': {'type': 'dashed'},
                    'data': [
                        {
                            'xAxis': r,
                            'lineStyle': {'color': GREEN, 'width': 1.5, 'type': 'dashed'},
                            'label': {'formatter': f'Range={r:.1f} m',
                                      'color': GREEN, 'fontSize': 11},
                        },
                        {
                            'yAxis': s + n,
                            'lineStyle': {'color': RED, 'width': 1.2, 'type': 'dotted'},
                            'label': {'formatter': f'Sill={s+n:.4f}',
                                      'color': RED, 'fontSize': 10},
                        },
                    ],
                },
            },
        ],
    }


# ── 2. Single layer-metric panel ──────────────────────────────────────────────

def layer_metric_echart(nl: list, vals: list, label: str, opt_n: int,
                        color: str, threshold: float = 0.85) -> dict:
    opt_idx = nl.index(opt_n) if opt_n in nl else None
    mark_pt = [{
        'coord': [opt_n, vals[opt_idx]],
        'symbol': 'pin', 'symbolSize': 18,
        'itemStyle': {'color': GOLD},
        'label': {'show': True, 'formatter': f'N={opt_n}',
                  'color': GOLD, 'fontSize': 10, 'position': 'top'},
    }] if opt_idx is not None else []

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
                    {'offset': 0, 'color': color.replace('#', 'rgba(') + ',0.18)' if color.startswith('#') else color},
                    {'offset': 1, 'color': 'rgba(0,0,0,0)'},
                ],
            }},
            'markLine': {
                'silent': True,
                'symbol': ['none', 'none'],
                'data': [{
                    'yAxis': threshold,
                    'lineStyle': {'color': 'rgba(255,255,255,0.22)', 'type': 'dotted'},
                    'label': {'formatter': f'{threshold:.0%}', 'color': TEXT2, 'fontSize': 10},
                }],
            },
            'markPoint': {'data': mark_pt},
        }],
    }


# ── 3. Cross plot ─────────────────────────────────────────────────────────────

def crossplot_echart(df_all: pd.DataFrame, x_prop: str, y_prop: str,
                     color_by: str = 'FACIES') -> dict:
    use_log_y = (y_prop == 'PERM')
    use_log_x = (x_prop == 'PERM')
    series, legend_data = [], []

    if color_by == 'FACIES' and 'FACIES' in df_all.columns:
        groups = sorted(df_all['FACIES'].unique())
        for g in groups:
            sub = df_all[df_all['FACIES'] == g]
            xv, yv = sub[x_prop].values, sub[y_prop].values
            data = [[float(x), float(y)] for x, y in zip(xv, yv)
                    if np.isfinite(x) and np.isfinite(y) and (y > 0 if use_log_y else True)]
            name = FACIES_NAMES.get(int(g), str(g))
            series.append({
                'type': 'scatter', 'name': name, 'data': data[:4000],
                'itemStyle': {'color': FACIES_COLORS.get(int(g), '#888'), 'opacity': 0.65},
                'symbolSize': 3, 'large': True, 'largeThreshold': 600,
            })
            legend_data.append(name)
    else:
        zones = sorted(df_all['ZONE'].unique()) if 'ZONE' in df_all.columns else []
        for z in zones:
            sub = df_all[df_all['ZONE'] == z]
            xv, yv = sub[x_prop].values, sub[y_prop].values
            data = [[float(x), float(y)] for x, y in zip(xv, yv)
                    if np.isfinite(x) and np.isfinite(y) and (y > 0 if use_log_y else True)]
            name = z.replace('_', ' ')
            series.append({
                'type': 'scatter', 'name': name, 'data': data[:4000],
                'itemStyle': {'color': ZONE_COLORS.get(z, '#888'), 'opacity': 0.65},
                'symbolSize': 3, 'large': True,
            })
            legend_data.append(name)

    return {
        'backgroundColor': BG,
        'tooltip': EC_TOOLTIP,
        'legend': {**EC_LEGEND, 'data': legend_data, 'type': 'scroll', 'top': 4},
        'xAxis': _axis(x_prop, log=use_log_x),
        'yAxis': _axis(y_prop, log=use_log_y),
        'grid': EC_GRID,
        'series': series,
        'dataZoom': [
            {'type': 'inside', 'xAxisIndex': 0},
            {'type': 'inside', 'yAxisIndex': 0},
            {'type': 'slider', 'xAxisIndex': 0, 'height': 16, 'bottom': 4,
             'textStyle': {'color': TEXT2},
             'fillerColor': 'rgba(59,130,246,0.15)',
             'borderColor': BORDER},
        ],
    }


# ── 4. Marginal gain bar chart ────────────────────────────────────────────────

def marginal_echart(results: pd.DataFrame) -> dict:
    opt_n = int(results.loc[results['optimal'], 'n_layers'].iloc[0]) \
            if results['optimal'].any() else 1
    nl = results['n_layers'].values.tolist()
    mg = results['marginal_gain'].values.tolist()
    bar_data = [
        {'value': [int(n), round(float(v), 5)],
         'itemStyle': {'color': GOLD if n == opt_n else BLUE, 'opacity': 0.85}}
        for n, v in zip(nl, mg)
    ]
    return {
        'backgroundColor': BG,
        'tooltip': {**EC_TOOLTIP, 'trigger': 'axis'},
        'xAxis': _axis('Number of Layers'),
        'yAxis': _axis('Marginal score gain'),
        'grid': dict(left='65', right='25', top='30', bottom='50', containLabel=True),
        'series': [{
            'type': 'bar', 'name': 'Marginal Gain',
            'data': bar_data,
            'markLine': {
                'silent': True, 'symbol': ['none', 'none'],
                'data': [{
                    'xAxis': opt_n,
                    'lineStyle': {'color': GOLD, 'type': 'dashed', 'width': 2},
                    'label': {'formatter': f'Opt N={opt_n}', 'color': GOLD, 'fontSize': 11},
                }],
            },
        }],
    }


# ── 5. Grid search heatmap ────────────────────────────────────────────────────

def grid_search_heatmap(all_results: dict) -> dict:
    """ECharts heatmap — Y = zones, X = N layers, colour = combined_score.

    all_results: {zone_name: pd.DataFrame from optimize_layers()}
    """
    zones = list(all_results.keys())
    if not zones:
        return {'backgroundColor': BG, 'series': []}

    max_n = max(int(df['n_layers'].max()) for df in all_results.values() if not df.empty)
    n_vals = list(range(1, max_n + 1))

    # heat data: [[n_idx, zone_idx, score], ...]
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
            heat_data.append([n, zi, round(s, 4)])
            if n == opt_n:
                star_data.append({'value': [n, zi], 'name': zname})

    return {
        'backgroundColor': BG,
        'tooltip': {
            'trigger': 'item',
            'backgroundColor': '#1a2332',
            'borderColor': BORDER,
            'textStyle': {'color': TEXT, 'fontSize': 11},
            'formatter': 'N layers: {b[0]}<br/>Zone: {b[1]}<br/>Score: {c}',
        },
        'grid': {'left': '120', 'right': '80', 'top': '40', 'bottom': '60', 'containLabel': False},
        'xAxis': {
            'type': 'category',
            'data': n_vals,
            'name': 'N Layers',
            'nameLocation': 'middle',
            'nameGap': 30,
            'nameTextStyle': {'color': TEXT2, 'fontSize': 11},
            'axisLabel': {'color': TEXT2, 'fontSize': 9, 'interval': 4},
            'axisLine': {'lineStyle': {'color': BORDER}},
            'splitLine': {'lineStyle': {'color': GRID_C, 'type': 'dashed'}},
        },
        'yAxis': {
            'type': 'category',
            'data': [z.replace('_', ' ') for z in zones],
            'axisLabel': {'color': TEXT2, 'fontSize': 10},
            'axisLine': {'lineStyle': {'color': BORDER}},
        },
        'visualMap': {
            'min': score_min,
            'max': score_max,
            'calculable': True,
            'orient': 'vertical',
            'right': 8,
            'top': '15%',
            'textStyle': {'color': TEXT2, 'fontSize': 9},
            'inRange': {'color': ['#0a1628', '#1e3a5f', '#3b82f6', '#10b981', '#f59e0b', '#ffd700']},
        },
        'series': [
            {
                'type': 'heatmap',
                'name': 'Score',
                'data': [[d[0] - 1, d[1], d[2]] for d in heat_data],  # x=index into category
                'label': {'show': False},
                'emphasis': {'itemStyle': {'shadowBlur': 8, 'shadowColor': GOLD}},
            },
            {
                'type': 'scatter',
                'name': 'Optimal N',
                'data': [{'value': [d['value'][0] - 1, d['value'][1]],
                           'label': {'show': True, 'formatter': f'N={d["value"][0]}',
                                     'color': GOLD, 'fontSize': 9, 'position': 'top'}}
                          for d in star_data],
                'symbol': 'pin',
                'symbolSize': 18,
                'itemStyle': {'color': GOLD},
                'zlevel': 2,
            },
        ],
    }


# ── 6. Stats table rows ───────────────────────────────────────────────────────

def stats_rows(results: pd.DataFrame) -> List[dict]:
    opt = results[results['optimal']].iloc[0] if results['optimal'].any() else results.iloc[-1]
    def badge(v, t): return '✓ Good' if v >= t else '⚠ Low'
    return [
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
