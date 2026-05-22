"""All Plotly figure generators for the reservoir modelling tool."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, List

from .analysis import (
    compute_variogram, fit_variogram, optimize_layers, upscale_zone,
    lorenz_coefficient, dykstra_parsons_v, recommend_from_variogram,
)

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------
BG      = '#0a0e1a'
PANEL   = '#111827'
GRID    = '#1e2d42'
TEXT    = '#e2e8f0'
TEXT2   = '#94a3b8'
BORDER  = '#2d3a4f'
BLUE    = '#3b82f6'
GREEN   = '#10b981'
AMBER   = '#f59e0b'
RED     = '#ef4444'
PURPLE  = '#8b5cf6'
CYAN    = '#06b6d4'

ZONE_COLORS = {
    'Overburden':   '#6b7280',
    'Hugin_Upper':  '#f59e0b',
    'Hugin_Middle': '#10b981',
    'Hugin_Lower':  '#3b82f6',
    'Basement':     '#9ca3af',
}

FACIES_COLORS = {
    0: '#5c3317',   # Shale – dark brown
    1: '#b8860b',   # Sandy Shale – dark goldenrod
    2: '#daa520',   # Tight Sand – goldenrod
    3: '#7cfc00',   # Good Sand – lawn green
    4: '#00e676',   # Excellent Sand – bright green
}

FACIES_COLORSCALE = [
    [0.00, '#5c3317'],
    [0.25, '#b8860b'],
    [0.50, '#daa520'],
    [0.75, '#7cfc00'],
    [1.00, '#00e676'],
]


def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=11),
        margin=dict(l=50, r=20, t=50, b=40),
        **kwargs,
    )


def _axis_style(**kwargs) -> dict:
    return dict(
        gridcolor=GRID,
        linecolor=BORDER,
        tickcolor=TEXT2,
        tickfont=dict(color=TEXT2, size=10),
        title_font=dict(color=TEXT, size=11),
        showgrid=True,
        zeroline=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Well Log Figure
# ---------------------------------------------------------------------------

def well_log_figure(df: pd.DataFrame, tops_df: pd.DataFrame, well_name: str) -> go.Figure:
    """6-track well log display (GR, PHIE, PERM, SW, NPHI/RHOB, Facies)."""
    if df is None or df.empty:
        return _empty_fig(f'No data for {well_name}')

    depth = df['DEPTH'].values
    gr    = df['GR'].values
    phie  = df['PHIE'].values
    perm  = df['PERM'].values
    sw    = df['SW'].values
    nphi  = df['NPHI'].values
    rhob  = df['RHOB'].values
    facies = df['FACIES'].values.astype(int)

    col_titles = ['GR (API)', 'PHIE', 'log K (mD)', 'Sw', 'NPHI / RHOB*', 'Facies']
    # Pre-compute for scale labels
    gr_shifted_arr = gr - 65.0
    log_k_arr      = np.log10(np.clip(perm, 1e-4, None))

    fig = make_subplots(
        rows=1, cols=6,
        shared_yaxes=True,
        horizontal_spacing=0.006,
        column_widths=[0.18, 0.15, 0.17, 0.15, 0.18, 0.17],
        subplot_titles=col_titles,
    )

    # -- Track 1: GR --
    gr_shifted = gr_shifted_arr   # already computed above
    fig.add_trace(go.Scatter(
        x=np.where(gr_shifted < 0, gr_shifted, 0), y=depth,
        fill='tozerox', fillcolor='rgba(255,200,0,0.30)',
        line=dict(width=0), name='Sand', showlegend=False, mode='lines',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=np.where(gr_shifted > 0, gr_shifted, 0), y=depth,
        fill='tozerox', fillcolor='rgba(160,160,160,0.20)',
        line=dict(width=0), name='Shale', showlegend=False, mode='lines',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=gr_shifted, y=depth,
        line=dict(color='#d1d5db', width=1.2), name='GR', showlegend=False,
    ), row=1, col=1)

    # -- Track 2: PHIE --
    fig.add_trace(go.Scatter(
        x=phie, y=depth,
        fill='tozerox', fillcolor='rgba(16,185,129,0.30)',
        line=dict(color=GREEN, width=1.4), name='PHIE', showlegend=False,
    ), row=1, col=2)

    # -- Track 3: PERM (log scale) --
    log_k = log_k_arr   # already computed above
    fig.add_trace(go.Scatter(
        x=log_k, y=depth,
        line=dict(color='#cd853f', width=1.4), name='PERM', showlegend=False,
    ), row=1, col=3)

    # -- Track 4: SW --
    fig.add_trace(go.Scatter(
        x=sw, y=depth,
        fill='tozerox', fillcolor='rgba(59,130,246,0.25)',
        line=dict(color=BLUE, width=1.4), name='Sw', showlegend=False,
    ), row=1, col=4)

    # -- Track 5: NPHI + RHOB normalised to same 0-0.45 scale --
    rhob_norm = (rhob - 1.95) / (2.88 - 1.95) * 0.45
    fig.add_trace(go.Scatter(
        x=nphi, y=depth,
        line=dict(color=BLUE, width=1.3, dash='solid'), name='NPHI', showlegend=False,
    ), row=1, col=5)
    fig.add_trace(go.Scatter(
        x=rhob_norm, y=depth,
        line=dict(color=RED, width=1.3, dash='solid'), name='RHOB*', showlegend=False,
    ), row=1, col=5)
    # Crossover fill (gas indicator: NPHI > RHOB*)
    fig.add_trace(go.Scatter(
        x=np.where(nphi > rhob_norm, nphi,  rhob_norm), y=depth,
        fill=None, mode='lines', line=dict(width=0), showlegend=False,
    ), row=1, col=5)

    # -- Track 6: Facies heatmap --
    fig.add_trace(go.Heatmap(
        z=facies.reshape(-1, 1),
        y=depth, x=[0, 1],
        colorscale=FACIES_COLORSCALE,
        zmin=0, zmax=4,
        showscale=False,
        hovertemplate='Depth: %{y:.1f} m<br>Facies: %{z}<extra></extra>',
    ), row=1, col=6)

    # -- Scale labels at top of each track (min | max in track header) --
    track_scales = [
        (1, 'x domain',  f'{int(np.nanmin(gr_shifted))}'  , f'{int(np.nanmax(gr_shifted))}'),
        (2, 'x2 domain', '0',                              f'{np.nanmax(phie):.3f}'),
        (3, 'x3 domain', f'{np.nanmin(log_k):.1f}',        f'{np.nanmax(log_k):.1f}'),
        (4, 'x4 domain', '0',                              '1.0'),
        (5, 'x5 domain', '0',                              '0.45'),
    ]
    for col, xref_k, vmin, vmax in track_scales:
        for xpos, val, anchor in [(0.0, vmin, 'left'), (1.0, vmax, 'right')]:
            fig.add_annotation(
                text=f'<b>{val}</b>',
                xref=xref_k, yref='y domain',
                x=xpos, y=1.02,
                xanchor=anchor, yanchor='bottom',
                showarrow=False,
                font=dict(color=TEXT2, size=8),
                bgcolor=PANEL,
                borderpad=1,
            )

    # -- Formation tops --
    if tops_df is not None and len(tops_df) > 0:
        for _, row in tops_df.iterrows():
            zname = row['ZONE']
            top   = float(row['TOP'])
            clr   = ZONE_COLORS.get(zname, '#ffffff')
            fig.add_hline(
                y=top,
                line=dict(color=clr, width=1.5, dash='dash'),
                annotation_text=zname.replace('_', ' '),
                annotation_position='bottom right',
                annotation_font=dict(size=9, color=clr),
            )

    # -- Axis styles --
    y_ax = _axis_style(autorange='reversed', title_text='Depth (m TVD)', showticklabels=True)
    for c in range(1, 7):
        if c == 1:
            fig.update_yaxes(**y_ax, row=1, col=c)
        else:
            fig.update_yaxes(autorange='reversed', showticklabels=False,
                             gridcolor=GRID, row=1, col=c)

    x_titles = ['GR − 65 (API)', 'PHIE', 'log₁₀ K', 'Sw', 'NPHI / RHOB*', '']
    for c, xt in enumerate(x_titles, 1):
        fig.update_xaxes(title_text=xt, **_axis_style(), row=1, col=c)
    fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=6)

    fig.update_layout(
        **_base_layout(height=720),
        title=dict(text=f'<b>Well: {well_name}</b>', font=dict(size=15, color=TEXT)),
        showlegend=False,
    )
    # Style subplot titles (track headers)
    for ann in fig.layout.annotations:
        ann.font.update(color=AMBER, size=11)
        ann.bgcolor = PANEL

    return fig


# ---------------------------------------------------------------------------
# 2. Well Section / Correlation
# ---------------------------------------------------------------------------

def well_section_figure(wells: Dict[str, pd.DataFrame], tops: pd.DataFrame,
                        selected_wells: Optional[List[str]] = None) -> go.Figure:
    """Side-by-side PHIE tracks with formation-top correlation lines."""
    all_names = list(wells.keys())
    names = selected_wells if selected_wells else all_names
    names = [n for n in names if n in wells]
    if not names:
        return _empty_fig('No wells selected')

    n_wells = len(names)
    fig = make_subplots(
        rows=1, cols=n_wells,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        subplot_titles=names,
    )

    depth_min, depth_max = 9999, 0
    for wi, wname in enumerate(names, 1):
        df = wells[wname]
        depth = df['DEPTH'].values
        phie  = df['PHIE'].values
        ntg   = df['NTG'].values
        facies = df['FACIES'].values.astype(int)

        depth_min = min(depth_min, depth.min())
        depth_max = max(depth_max, depth.max())

        # PHIE fill
        fig.add_trace(go.Scatter(
            x=phie, y=depth,
            fill='tozerox', fillcolor='rgba(16,185,129,0.35)',
            line=dict(color=GREEN, width=1.2), name='PHIE', showlegend=False,
        ), row=1, col=wi)

        # NTG as thin background fill
        fig.add_trace(go.Scatter(
            x=ntg * 0.45, y=depth,
            fill='tozerox', fillcolor='rgba(245,158,11,0.12)',
            line=dict(width=0), name='NTG', showlegend=False,
        ), row=1, col=wi)

        # Zone colour bands from tops
        wtops = tops[tops['WELL'] == wname] if tops is not None else pd.DataFrame()
        for _, tr in wtops.iterrows():
            zname = tr['ZONE']
            clr   = ZONE_COLORS.get(zname, '#ffffff')
            fig.add_hline(
                y=float(tr['TOP']),
                line=dict(color=clr, width=1.2, dash='dash'),
                row=1, col=wi,
            )

    # Zone legend traces (for colour reference in legend)
    if tops is not None:
        for zname, zclr in ZONE_COLORS.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color=zclr, width=1.5, dash='dash'),
                name=zname.replace('_', ' '),
                showlegend=True,
            ), row=1, col=1)

    for c in range(1, n_wells + 1):
        ya = _axis_style(autorange='reversed')
        if c == 1:
            ya['title_text'] = 'Depth (m)'
        else:
            ya['showticklabels'] = False
        fig.update_yaxes(**ya, row=1, col=c)
        fig.update_xaxes(title_text='PHIE', **_axis_style(), row=1, col=c)

    for ann in fig.layout.annotations:
        ann.font.update(color=AMBER, size=11)

    fig.update_layout(
        **_base_layout(height=720),
        title=dict(text='<b>Well Correlation Section</b>', font=dict(size=15, color=TEXT)),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Cross Plot
# ---------------------------------------------------------------------------

PROP_RANGES = {
    'GR':   (0, 160),
    'PHIE': (0, 0.40),
    'PERM': None,        # log scale
    'SW':   (0, 1),
    'NTG':  (0, 1),
    'NPHI': (0, 0.50),
    'RHOB': (1.9, 2.9),
}
PROP_LABELS = {
    'GR': 'Gamma Ray (API)', 'PHIE': 'Effective Porosity',
    'PERM': 'Permeability (mD)', 'SW': 'Water Saturation',
    'NTG': 'Net-to-Gross', 'NPHI': 'Neutron Porosity', 'RHOB': 'Bulk Density (g/cc)',
}


def crossplot_figure(
    wells: Dict[str, pd.DataFrame],
    x_prop: str = 'PHIE',
    y_prop: str = 'PERM',
    color_by: str = 'FACIES',
    zone_filter: Optional[str] = None,
) -> go.Figure:
    """Interactive cross plot coloured by facies or zone."""
    frames = []
    for wname, df in wells.items():
        tmp = df.copy()
        tmp['WELL'] = wname
        if zone_filter and zone_filter != 'All':
            tmp = tmp[tmp['ZONE'] == zone_filter]
        frames.append(tmp)

    if not frames:
        return _empty_fig('No data')

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=[x_prop, y_prop])

    use_log_y = (y_prop == 'PERM')
    use_log_x = (x_prop == 'PERM')

    if color_by == 'FACIES':
        from .synthetic import FACIES_NAMES
        color_vals = all_df['FACIES'].astype(int)
        color_map  = {v: FACIES_COLORS[v] for v in sorted(FACIES_COLORS)}
        color_col  = color_vals.map(lambda v: FACIES_COLORS.get(v, '#888888'))
        legend_items = [(k, FACIES_NAMES[k], FACIES_COLORS[k]) for k in sorted(FACIES_NAMES)]
        color_title = 'Facies'
    else:
        zone_list = sorted(all_df['ZONE'].unique())
        zone_map  = {z: i for i, z in enumerate(zone_list)}
        color_col = all_df['ZONE'].map(lambda z: ZONE_COLORS.get(z, '#888888'))
        legend_items = [(i, z, ZONE_COLORS.get(z, '#888888')) for i, z in enumerate(zone_list)]
        color_title = 'Zone'

    fig = go.Figure()

    # One trace per colour group for legend
    if color_by == 'FACIES':
        from .synthetic import FACIES_NAMES
        groups = sorted(all_df['FACIES'].unique())
        for g in groups:
            sub = all_df[all_df['FACIES'] == g]
            x_vals = sub[x_prop].values
            y_vals = sub[y_prop].values if not use_log_y else np.log10(np.clip(sub[y_prop].values, 1e-4, None))
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers',
                marker=dict(color=FACIES_COLORS.get(g, '#888'), size=4, opacity=0.65),
                name=FACIES_NAMES.get(g, str(g)),
                customdata=sub[['DEPTH', 'WELL', 'SW', 'NTG']].values,
                hovertemplate=(f'Depth: %{{customdata[0]:.1f}} m<br>'
                               f'Well: %{{customdata[1]}}<br>'
                               f'{x_prop}: %{{x:.3f}}<br>'
                               f'{y_prop}: %{{y:.2f}}<extra></extra>'),
            ))
    else:
        for zname in sorted(all_df['ZONE'].unique()):
            sub = all_df[all_df['ZONE'] == zname]
            x_vals = sub[x_prop].values
            y_vals = sub[y_prop].values if not use_log_y else np.log10(np.clip(sub[y_prop].values, 1e-4, None))
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='markers',
                marker=dict(color=ZONE_COLORS.get(zname, '#888'), size=4, opacity=0.65),
                name=zname.replace('_', ' '),
            ))

    y_label = ('log₁₀ ' if use_log_y else '') + PROP_LABELS.get(y_prop, y_prop)
    x_label = ('log₁₀ ' if use_log_x else '') + PROP_LABELS.get(x_prop, x_prop)

    fig.update_layout(
        **_base_layout(height=520),
        title=dict(text=f'<b>Cross Plot: {x_prop} vs {y_prop}</b>  |  Colour: {color_title}',
                   font=dict(size=14, color=TEXT)),
        xaxis=_axis_style(title_text=x_label),
        yaxis=_axis_style(title_text=y_label),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT, size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. 3-D Well View
# ---------------------------------------------------------------------------

def figure_3d(wells: Dict[str, pd.DataFrame], tops: pd.DataFrame,
              color_prop: str = 'PHIE') -> go.Figure:
    """3-D well trajectories coloured by property + formation-top planes."""
    fig = go.Figure()

    well_colors = [BLUE, GREEN, AMBER, RED, PURPLE]

    for wi, (wname, df) in enumerate(wells.items()):
        x  = df['X'].values
        y  = df['Y'].values
        z  = -df['TVDSS'].values    # negate for depth = downward
        prop = df[color_prop].values if color_prop in df.columns else df['PHIE'].values

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(color=prop, colorscale='Viridis', width=4),
            marker=dict(size=1, color=prop, colorscale='Viridis', opacity=0.6),
            name=wname,
            hovertemplate=(f'<b>{wname}</b><br>X: %{{x:.0f}}<br>Y: %{{y:.0f}}<br>'
                           f'TVD: %{{customdata:.1f}} m<extra></extra>'),
            customdata=df['TVDSS'].values,
            showlegend=True,
        ))
        # Well label at top
        fig.add_trace(go.Scatter3d(
            x=[x[0]], y=[y[0]], z=[z[0]],
            mode='text',
            text=[wname],
            textfont=dict(color=well_colors[wi % len(well_colors)], size=12),
            showlegend=False,
        ))

    # Formation-top planes (thin horizontal markers)
    if tops is not None:
        for zname, zclr in ZONE_COLORS.items():
            zone_tops = tops[tops['ZONE'] == zname]
            if zone_tops.empty:
                continue
            avg_top = float(zone_tops['TOP'].mean())
            xs = [float(r['X']) for _, r in zone_tops.iterrows()]
            ys = [float(r['Y']) for _, r in zone_tops.iterrows()]
            if len(xs) >= 2:
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=[-avg_top] * len(xs),
                    mode='markers+lines',
                    line=dict(color=zclr, width=2, dash='dash'),
                    marker=dict(size=4, color=zclr, symbol='diamond'),
                    name=zname.replace('_', ' '),
                    showlegend=True,
                ))

    fig.update_layout(
        **_base_layout(height=620),
        title=dict(text=f'<b>3-D Well View</b>  |  Colour: {color_prop}',
                   font=dict(size=15, color=TEXT)),
        scene=dict(
            bgcolor=PANEL,
            xaxis=dict(title='Easting (m)', gridcolor=GRID, backgroundcolor=BG,
                       tickfont=dict(color=TEXT2)),
            yaxis=dict(title='Northing (m)', gridcolor=GRID, backgroundcolor=BG,
                       tickfont=dict(color=TEXT2)),
            zaxis=dict(title='−TVDSS (m)', gridcolor=GRID, backgroundcolor=BG,
                       tickfont=dict(color=TEXT2)),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1.2, z=0.8),
        ),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT, size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Variogram
# ---------------------------------------------------------------------------

def variogram_figure(df_zone: pd.DataFrame, prop: str, well_name: str,
                     zone_name: str, model_type: str = 'Spherical') -> go.Figure:
    """Experimental + fitted variogram for a single well / zone / property."""
    if df_zone is None or df_zone.empty or prop not in df_zone.columns:
        return _empty_fig('Insufficient data for variogram')

    z    = df_zone[prop].values
    dz   = float(df_zone['DEPTH'].diff().median())
    lags, gamma, npairs = compute_variogram(z, dz, n_lags=30)

    result = fit_variogram(lags, gamma, model_type)
    h_fit  = result['h_fit']
    g_fit  = result['g_fit']
    rng_v  = result['range']
    sill_v = result['sill']
    nug_v  = result['nugget']
    zone_thick = float(df_zone['DEPTH'].max() - df_zone['DEPTH'].min())
    rec_n = recommend_from_variogram(zone_thick, rng_v)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Variogram', 'Variance vs Depth Lag'],
                        column_widths=[0.65, 0.35])

    # Experimental variogram
    fig.add_trace(go.Scatter(
        x=lags, y=gamma,
        mode='markers',
        marker=dict(color=AMBER, size=7, symbol='circle'),
        name='Experimental',
        hovertemplate='Lag: %{x:.1f} m<br>γ: %{y:.5f}<extra></extra>',
    ), row=1, col=1)

    # Fitted model
    fig.add_trace(go.Scatter(
        x=h_fit, y=g_fit,
        mode='lines',
        line=dict(color=BLUE, width=2.5),
        name=f'{model_type} model',
    ), row=1, col=1)

    # Range line
    fig.add_vline(x=rng_v, line=dict(color=GREEN, width=1.5, dash='dash'),
                  row=1, col=1)
    # Sill line
    fig.add_hline(y=sill_v + nug_v, line=dict(color=RED, width=1.2, dash='dot'),
                  row=1, col=1)

    # Annotations
    fig.add_annotation(x=rng_v, y=sill_v * 0.5,
                       text=f'Range={rng_v:.1f} m',
                       font=dict(color=GREEN, size=11),
                       showarrow=False, xshift=8, row=1, col=1)

    # Pair count bar chart
    fig.add_trace(go.Bar(
        x=lags, y=npairs,
        marker=dict(color=PURPLE, opacity=0.6),
        name='# pairs', showlegend=True,
    ), row=1, col=2)

    fig.update_xaxes(title_text='Lag distance (m)', **_axis_style(), row=1, col=1)
    fig.update_yaxes(title_text='Semivariance γ(h)', **_axis_style(), row=1, col=1)
    fig.update_xaxes(title_text='Lag (m)', **_axis_style(), row=1, col=2)
    fig.update_yaxes(title_text='# Pairs', **_axis_style(), row=1, col=2)

    for ann in fig.layout.annotations:
        ann.font.update(color=AMBER, size=11)

    fig.update_layout(
        **_base_layout(height=440),
        title=dict(
            text=(f'<b>Variogram Analysis</b> — Well: {well_name}, Zone: {zone_name}, '
                  f'Property: {prop}<br>'
                  f'<span style="font-size:12px;color:{GREEN}">'
                  f'Nugget={nug_v:.4f}  Sill={sill_v:.4f}  Range={rng_v:.1f} m  '
                  f'→  Recommended min layers: {rec_n}</span>'),
            font=dict(size=13, color=TEXT),
        ),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT, size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Layer Optimisation Dashboard
# ---------------------------------------------------------------------------

def layer_analysis_figure(results: pd.DataFrame, zone_name: str) -> go.Figure:
    """4-panel dashboard of preservation metrics vs N layers."""
    if results is None or results.empty:
        return _empty_fig('Run layer analysis first')

    opt_n  = int(results.loc[results['optimal'], 'n_layers'].values[0]) if results['optimal'].any() else 1
    vp_n   = int(results['vp_threshold_n'].iloc[0])
    nl     = results['n_layers'].values

    metrics = [
        ('variance_preservation', 'Variance Preservation',   BLUE,   '≥ 85% target'),
        ('lorenz_preservation',   'Lorenz Coefficient Pres.', GREEN,  '≥ 85% target'),
        ('dp_preservation',       'Dykstra-Parsons Pres.',   AMBER,  '≥ 80% target'),
        ('facies_coverage',       'Facies Coverage',         PURPLE, '= 100% target'),
    ]
    thresholds = [0.85, 0.85, 0.80, 1.00]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[m[1] for m in metrics],
                        vertical_spacing=0.14,
                        horizontal_spacing=0.10)

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (col_key, label, clr, tgt), (row, col), thresh in zip(metrics, positions, thresholds):
        vals = results[col_key].values

        fig.add_trace(go.Scatter(
            x=nl, y=vals,
            mode='lines+markers',
            line=dict(color=clr, width=2.5),
            marker=dict(size=5, color=clr),
            name=label, showlegend=False,
            hovertemplate=f'N={"%{x}"}  {label}={"%{y:.3f}"}<extra></extra>',
        ), row=row, col=col)

        # Threshold line
        fig.add_hline(y=thresh, line=dict(color='rgba(255,255,255,0.25)', width=1, dash='dot'),
                      row=row, col=col)

        # Optimal N marker
        opt_val = results.loc[results['n_layers'] == opt_n, col_key]
        if len(opt_val):
            fig.add_trace(go.Scatter(
                x=[opt_n], y=[float(opt_val.values[0])],
                mode='markers',
                marker=dict(symbol='star', size=14, color='#ffd700',
                            line=dict(width=1, color='black')),
                name=f'Optimal N={opt_n}', showlegend=(row == 1 and col == 1),
            ), row=row, col=col)

    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(title_text='Number of Layers', **_axis_style(), row=row, col=col)
            fig.update_yaxes(range=[0, 1.05], **_axis_style(), row=row, col=col)

    for ann in fig.layout.annotations:
        ann.font.update(color=AMBER, size=11)

    fig.update_layout(
        **_base_layout(height=560),
        title=dict(
            text=(f'<b>Layer Optimisation — Zone: {zone_name}</b><br>'
                  f'<span style="font-size:12px;color:#ffd700">'
                  f'★ Recommended: {opt_n} layers  |  '
                  f'Variance threshold (85%): {vp_n} layers</span>'),
            font=dict(size=14, color=TEXT),
        ),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT, size=10),
                    x=0.78, y=0.98),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Log vs Upscaled Comparison
# ---------------------------------------------------------------------------

def upscaled_comparison_figure(df_zone: pd.DataFrame, n_layers: int,
                                zone_name: str) -> go.Figure:
    """Side-by-side original log vs upscaled step functions."""
    if df_zone is None or df_zone.empty:
        return _empty_fig('No zone data')

    up = upscale_zone(df_zone, n_layers)
    if up.empty:
        return _empty_fig('Upscaling failed')

    props = [
        ('PHIE', GREEN,      'PHIE'),
        ('PERM', '#cd853f',  'PERM (log scale)'),
        ('SW',   BLUE,       'Sw'),
        ('NTG',  AMBER,      'NTG'),
    ]
    n_props = len(props)

    fig = make_subplots(rows=1, cols=n_props, shared_yaxes=True,
                        horizontal_spacing=0.01,
                        subplot_titles=[p[2] for p in props])

    depth_orig = df_zone['DEPTH'].values

    for ci, (prop, clr, label) in enumerate(props, 1):
        if prop not in df_zone.columns:
            continue

        orig_vals = df_zone[prop].values
        use_log   = (prop == 'PERM')

        x_orig = np.log10(np.clip(orig_vals, 1e-4, None)) if use_log else orig_vals
        fig.add_trace(go.Scatter(
            x=x_orig, y=depth_orig,
            mode='lines',
            line=dict(color=clr, width=1.2, dash='solid'),
            opacity=0.5,
            name=f'{prop} original', showlegend=(ci == 1),
        ), row=1, col=ci)

        # Build step function for upscaled
        step_x, step_y = [], []
        for _, ur in up.iterrows():
            v = float(ur[prop])
            xv = np.log10(max(v, 1e-4)) if use_log else v
            step_x += [xv, xv, xv]
            step_y += [float(ur['DEPTH_TOP']), float(ur['DEPTH_BASE']), None]

        fig.add_trace(go.Scatter(
            x=step_x, y=step_y,
            mode='lines',
            line=dict(color=clr, width=3, dash='solid'),
            name=f'{prop} upscaled ({n_layers}L)', showlegend=(ci == 1),
        ), row=1, col=ci)

    for c in range(1, n_props + 1):
        fig.update_xaxes(**_axis_style(), row=1, col=c)
        if c == 1:
            fig.update_yaxes(**_axis_style(autorange='reversed', title_text='Depth (m)'),
                             row=1, col=c)
        else:
            fig.update_yaxes(autorange='reversed', showticklabels=False,
                             gridcolor=GRID, row=1, col=c)

    for ann in fig.layout.annotations:
        ann.font.update(color=AMBER, size=11)

    # Compute preservation stats
    phie_vp = _variance_preservation(df_zone['PHIE'].values, up['PHIE'].values)

    fig.update_layout(
        **_base_layout(height=620),
        title=dict(
            text=(f'<b>Log vs Upscaled — Zone: {zone_name}  |  N = {n_layers} layers</b><br>'
                  f'<span style="font-size:12px;color:{GREEN}">'
                  f'PHIE Variance Preservation: {phie_vp:.1%}</span>'),
            font=dict(size=13, color=TEXT),
        ),
        legend=dict(bgcolor=PANEL, bordercolor=BORDER, font=dict(color=TEXT, size=10)),
    )
    return fig


def petrel_upscaled_allzones(
    wells: dict, zone_list: list, n_layers_dict: dict, well_name: str,
    tops_df: Optional[pd.DataFrame] = None,
    props_to_show: Optional[List[str]] = None,
) -> go.Figure:
    """Petrel-style upscaled display — per-zone upscaling, combined depth view."""
    df_well = wells.get(well_name, pd.DataFrame())
    if df_well.empty:
        return _empty_fig(f'No data for {well_name}')

    orig_frames, up_frames = [], []
    for zone in zone_list:
        zdf = df_well[df_well['ZONE'] == zone].copy()
        if zdf.empty:
            continue
        up = upscale_zone(zdf, n_layers_dict.get(zone, 10))
        if not up.empty:
            up_frames.append(up)
        zdf['_zone'] = zone
        orig_frames.append(zdf)

    if not orig_frames:
        return _empty_fig('No zone data')

    df_all  = pd.concat(orig_frames, ignore_index=True).sort_values('DEPTH')
    all_up  = pd.concat(up_frames,   ignore_index=True) if up_frames else pd.DataFrame()
    label   = ' + '.join(z.replace('_', ' ') for z in zone_list)
    return petrel_upscaled_figure(df_all, 0, label, _multi=True,
                                  tops_df=tops_df, _pre_up=all_up,
                                  props_to_show=props_to_show)


def petrel_upscaled_figure(df_zone: pd.DataFrame, n_layers: int,
                           zone_name: str, _multi: bool = False,
                           tops_df: Optional[pd.DataFrame] = None,
                           _pre_up: Optional[pd.DataFrame] = None,
                           props_to_show: Optional[List[str]] = None) -> go.Figure:
    """Petrel-style upscaled display: coloured cell blocks + original log overlay.

    Row 1 (thin): colourbar strip per property track.
    Row 2 (tall): heatmap cell blocks + thin black original log overlay.
    Rainbow (blue→cyan→green→yellow→red) colorscale per track.
    White horizontal grid lines at every layer boundary.
    """
    if df_zone is None or df_zone.empty:
        return _empty_fig('No zone data')

    # Use pre-built upscaled df (all-zones mode) or upscale from scratch
    if _pre_up is not None and not _pre_up.empty:
        up = _pre_up
        n_layers_actual = len(up)
    else:
        n_layers_actual = max(1, n_layers if not _multi else max(5, n_layers))
        up = upscale_zone(df_zone, n_layers_actual)
    if up.empty:
        return _empty_fig('Upscaling failed')

    _all_props = [
        ('PHIE', 'PHIE',      0.0,  0.35),
        ('PERM', 'PERM (mD)', 0.01, None),
        ('SW',   'Sw',        0.0,  1.0),
        ('NTG',  'NTG',       0.0,  1.0),
    ]
    props   = [p for p in _all_props if p[0] in props_to_show] if props_to_show else _all_props
    n_props = len(props)

    PETREL_CS = [
        [0.00, '#0000cc'],
        [0.16, '#00ccff'],
        [0.33, '#00ff88'],
        [0.50, '#ffff00'],
        [0.67, '#ff8800'],
        [0.83, '#ff2200'],
        [1.00, '#880000'],
    ]

    # 2 rows: thin colorbar strip (row1) + main data (row2)
    _prop_w = 0.96 / max(n_props, 1)
    col_widths = [0.04] + [_prop_w] * n_props
    row_heights = [0.06, 0.94]

    fig = make_subplots(
        rows=2, cols=n_props + 1,
        shared_yaxes=True,
        shared_xaxes=False,
        row_heights=row_heights,
        horizontal_spacing=0.004,
        vertical_spacing=0.01,
        column_widths=col_widths,
    )

    depth_orig = df_zone['DEPTH'].values

    # ── Depth ruler column (col 1, row 2) ─────────────────────────────────────
    d_min, d_max = float(depth_orig.min()), float(depth_orig.max())
    fig.add_trace(go.Scatter(x=[0, 0], y=[d_min, d_max],
                             mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'), row=2, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False,
                     zeroline=False, range=[0, 1], row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False,
                     zeroline=False, range=[0, 1], row=2, col=1)

    for ci, (prop, label, vmin, vmax) in enumerate(props, 2):
        if prop not in df_zone.columns:
            continue

        orig_vals = df_zone[prop].values
        use_log   = (prop == 'PERM')

        # Value range
        if use_log:
            orig_plot = np.log10(np.clip(orig_vals, 0.01, None))
            zmin_h    = float(np.nanmin(orig_plot))
            zmax_h    = float(np.nanmax(orig_plot))
        else:
            orig_plot = orig_vals.astype(float)
            zmin_h    = vmin if vmin is not None else float(np.nanmin(orig_vals))
            zmax_h    = vmax if vmax is not None else float(np.nanmax(orig_vals))
        if zmax_h <= zmin_h:
            zmax_h = zmin_h + 1e-6

        # ── Row 1: colorbar strip (gradient bar across full property range) ───
        strip_n  = 50
        strip_z  = np.linspace(zmin_h, zmax_h, strip_n).reshape(1, -1)
        strip_x  = np.linspace(0, 1, strip_n)
        fig.add_trace(go.Heatmap(
            z=strip_z,
            x=strip_x,
            y=[0],
            zmin=zmin_h, zmax=zmax_h,
            colorscale=PETREL_CS,
            zsmooth=False,
            showscale=False,
            hoverinfo='skip',
            name=label,
        ), row=1, col=ci)
        fig.update_xaxes(showticklabels=False, showgrid=False,
                         zeroline=False, range=[0, 1], row=1, col=ci)

        # ── Row 2: heatmap cell blocks ─────────────────────────────────────────
        y_centers = up['DEPTH_MID'].values.tolist()
        z_vals    = []
        for _, ur in up.iterrows():
            v = float(ur[prop])
            z_vals.append(np.log10(max(v, 0.01)) if use_log else v)

        z_arr = np.array([[v, v] for v in z_vals])
        layer_bounds = (list(up['DEPTH_TOP'].values)
                        + [float(up['DEPTH_BASE'].values[-1])])

        fig.add_trace(go.Heatmap(
            z=z_arr,
            y=y_centers,
            x=[0.1, 0.9],
            zmin=zmin_h, zmax=zmax_h,
            colorscale=PETREL_CS,
            zsmooth=False,
            showscale=False,
            hovertemplate=f'{label}: %{{z:.4f}}<br>Depth: %{{y:.1f}} m<extra></extra>',
        ), row=2, col=ci)

        # White layer-boundary lines
        for bd in layer_bounds:
            fig.add_hline(y=bd, line=dict(color='white', width=0.7),
                          row=2, col=ci)

        # Original log overlay (thin black)
        x_norm = (orig_plot - zmin_h) / (zmax_h - zmin_h)
        fig.add_trace(go.Scatter(
            x=x_norm, y=depth_orig,
            mode='lines',
            line=dict(color='black', width=1.1),
            showlegend=False, hoverinfo='skip',
        ), row=2, col=ci)

        fig.update_xaxes(showticklabels=False, showgrid=False,
                         zeroline=False, range=[0, 1], row=2, col=ci)

    # ── Formation tops overlay on main data rows ─────────────────────────────
    if tops_df is not None and len(tops_df) > 0:
        shown_labels = set()
        for _, tr in tops_df.iterrows():
            zname = tr['ZONE']
            top_v = float(tr['TOP'])
            clr   = ZONE_COLORS.get(zname, '#ffffff')
            for ci in range(2, n_props + 2):
                kw = {}
                if ci == 2 and zname not in shown_labels:
                    kw = dict(annotation_text=f'<b>{zname.replace("_"," ")}</b>',
                              annotation_position='top left',
                              annotation_font=dict(size=8, color=clr))
                    shown_labels.add(zname)
                fig.add_hline(y=top_v, row=2, col=ci,
                              line=dict(color=clr, width=1.8, dash='dash'), **kw)

    # ── Y axes ──────────────────────────────────────────────────────────────
    fig.update_yaxes(autorange='reversed', tickfont=dict(color=TEXT2, size=10),
                     gridcolor=GRID, linecolor=BORDER, zeroline=False,
                     title_text='Depth (m)', row=2, col=1)
    for ci in range(2, n_props + 2):
        fig.update_yaxes(autorange='reversed', showticklabels=False,
                         gridcolor=GRID, row=2, col=ci)
    # Row 1 y axes minimal
    for ci in range(1, n_props + 2):
        fig.update_yaxes(showticklabels=False, showgrid=False,
                         zeroline=False, row=1, col=ci)

    phie_vp = _variance_preservation(df_zone['PHIE'].values, up['PHIE'].values)

    fig.update_layout(
        **_base_layout(height=720),
        title=dict(
            text=(f'<b>Log vs Upscaled (Petrel Style) — Zone: {zone_name}  '
                  f'|  N = {n_layers_actual} layers</b><br>'
                  f'<span style="font-size:12px;color:{GREEN}">'
                  f'PHIE Variance Preservation: {phie_vp:.1%}</span>'),
            font=dict(size=13, color=TEXT),
        ),
        showlegend=False,
    )
    return fig


def _variance_preservation(orig: np.ndarray, up: np.ndarray) -> float:
    v_orig = np.var(orig)
    if v_orig < 1e-15:
        return 1.0
    return min(1.0, np.var(up) / v_orig)


# ---------------------------------------------------------------------------
# 8. Summary Stats Cards (table figure)
# ---------------------------------------------------------------------------

def summary_table_figure(results: pd.DataFrame) -> go.Figure:
    """Table showing key metrics at optimal N."""
    if results is None or results.empty:
        return _empty_fig('No results')

    opt = results[results['optimal']].iloc[0] if results['optimal'].any() else results.iloc[-1]

    headers = ['Metric', 'Value']
    rows = [
        ['Recommended N layers',   str(int(opt['n_layers']))],
        ['VP threshold N (≥85%)',  str(int(opt['vp_threshold_n']))],
        ['Variance Preservation',  f"{opt['variance_preservation']:.1%}"],
        ['Lorenz Preservation',    f"{opt['lorenz_preservation']:.1%}"],
        ['DP Preservation',        f"{opt['dp_preservation']:.1%}"],
        ['Facies Coverage',        f"{opt['facies_coverage']:.1%}"],
        ['Combined Score',         f"{opt['combined_score']:.3f}"],
        ['Lorenz Coeff (orig)',    f"{opt['lorenz_coeff_orig']:.3f}"],
        ['DP Coeff (orig)',        f"{opt['dp_coeff_orig']:.3f}"],
    ]

    fig = go.Figure(go.Table(
        header=dict(
            values=['<b>Metric</b>', '<b>Value</b>'],
            fill_color=GRID,
            font=dict(color=AMBER, size=12),
            align='left',
            height=30,
        ),
        cells=dict(
            values=[[r[0] for r in rows], [r[1] for r in rows]],
            fill_color=[PANEL, BG],
            font=dict(color=[TEXT, GREEN], size=11),
            align='left',
            height=26,
        ),
    ))
    fig.update_layout(**_base_layout(height=340, margin=dict(l=10, r=10, t=30, b=10)))
    return fig


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _empty_fig(msg: str = 'No data') -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref='paper', yref='paper',
                       showarrow=False, font=dict(color=TEXT2, size=16))
    fig.update_layout(**_base_layout(height=400))
    return fig
