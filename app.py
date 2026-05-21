"""LayerOptimum Pro — Reservoir Layer Optimisation Tool."""

import json
import base64
import io
import numpy as np
import pandas as pd
import lasio

import dash
from dash import dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

from src.synthetic import (
    generate_synthetic_wells, generate_formation_tops,
    WELL_NAMES, RESERVOIR_ZONES, FACIES_NAMES,
)
from src.analysis import optimize_layers, compute_variogram, fit_variogram
from src.figures import (
    well_log_figure, well_section_figure, crossplot_figure,
    figure_3d, variogram_figure, layer_analysis_figure,
    upscaled_comparison_figure, summary_table_figure, _empty_fig,
    ZONE_COLORS, FACIES_COLORS,
)

# ---------------------------------------------------------------------------
# Startup data
# ---------------------------------------------------------------------------
WELLS = generate_synthetic_wells()
TOPS  = generate_formation_tops(WELLS)

ALL_ZONES    = ['All'] + list(ZONE_COLORS.keys())
RES_ZONES    = RESERVOIR_ZONES
PROPS        = ['GR', 'PHIE', 'PERM', 'SW', 'NTG', 'NPHI', 'RHOB']
COLOR_BY_OPT = ['FACIES', 'ZONE', 'WELL']

# Pre-serialize for dcc.Store (one JSON per well)
def _serialize_wells(wells):
    return {k: v.to_json(orient='split', date_format='iso') for k, v in wells.items()}

def _deserialize_wells(data: dict):
    return {k: pd.read_json(io.StringIO(v), orient='split') for k, v in data.items()}

WELLS_INIT = _serialize_wells(WELLS)
TOPS_INIT  = TOPS.to_json(orient='split')

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title='LayerOptimum Pro',
)
server = app.server

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _dropdown(id_, options, value, extra_style=None, **kwargs):
    base_style = {'background': '#1a2332', 'color': '#e2e8f0',
                  'border': '1px solid #2d3a4f', 'fontSize': '11px'}
    if extra_style:
        base_style = {**base_style, **extra_style}
    return dcc.Dropdown(
        id=id_,
        options=[{'label': o, 'value': o} for o in options],
        value=value,
        clearable=False,
        style=base_style,
        **kwargs,
    )


def sidebar():
    return html.Div(className='sidebar', children=[

        # Well selector
        html.Div(className='sidebar-section', children=[
            html.Span('Wells', className='sidebar-label'),
            dcc.Checklist(
                id='well-checklist',
                options=[{'label': f'  {w}', 'value': w} for w in WELL_NAMES],
                value=[WELL_NAMES[0]],
                inputStyle={'margin-right': '6px', 'accentColor': '#3b82f6'},
                labelStyle={'display': 'block', 'color': '#94a3b8',
                            'fontSize': '11px', 'marginBottom': '4px'},
            ),
            html.Div(style={'display': 'flex', 'gap': '4px', 'marginTop': '6px'}, children=[
                html.Button('All', id='btn-all-wells', n_clicks=0,
                            style={'flex': 1, 'fontSize': '10px', 'padding': '3px',
                                   'background': '#1a2332', 'color': '#94a3b8',
                                   'border': '1px solid #2d3a4f', 'borderRadius': '3px',
                                   'cursor': 'pointer'}),
                html.Button('None', id='btn-no-wells', n_clicks=0,
                            style={'flex': 1, 'fontSize': '10px', 'padding': '3px',
                                   'background': '#1a2332', 'color': '#94a3b8',
                                   'border': '1px solid #2d3a4f', 'borderRadius': '3px',
                                   'cursor': 'pointer'}),
            ]),
        ]),

        # Active well for log display
        html.Div(className='sidebar-section', children=[
            html.Span('Log Display Well', className='sidebar-label'),
            _dropdown('well-log-selector', WELL_NAMES, WELL_NAMES[0]),
        ]),

        html.Div(className='divider'),

        # Zone selector
        html.Div(className='sidebar-section', children=[
            html.Span('Analysis Zone', className='sidebar-label'),
            _dropdown('zone-selector', RES_ZONES, RES_ZONES[-1]),
        ]),

        # Property for variogram / crossplot
        html.Div(className='sidebar-section', children=[
            html.Span('Property', className='sidebar-label'),
            _dropdown('prop-selector', PROPS, 'PHIE'),
        ]),

        html.Div(className='divider'),

        # N layers slider
        html.Div(className='sidebar-section', children=[
            html.Span('N Layers (upscaling)', className='sidebar-label'),
            dcc.Slider(id='nlayers-slider', min=1, max=40, step=1, value=10,
                       marks={1: '1', 10: '10', 20: '20', 30: '30', 40: '40'},
                       tooltip={'placement': 'bottom', 'always_visible': True},
                       updatemode='drag'),
        ]),

        # Variogram model
        html.Div(className='sidebar-section', children=[
            html.Span('Variogram Model', className='sidebar-label'),
            dcc.RadioItems(
                id='vario-model',
                options=[{'label': f'  {m}', 'value': m}
                         for m in ['Spherical', 'Exponential', 'Gaussian']],
                value='Spherical',
                inputStyle={'margin-right': '6px', 'accentColor': '#f59e0b'},
                labelStyle={'display': 'block', 'color': '#94a3b8',
                            'fontSize': '11px', 'marginBottom': '4px'},
            ),
        ]),

        # Max layers for optimisation
        html.Div(className='sidebar-section', children=[
            html.Span('Max Layers (analysis)', className='sidebar-label'),
            dcc.Slider(id='maxlayers-slider', min=5, max=50, step=5, value=35,
                       marks={5: '5', 20: '20', 35: '35', 50: '50'},
                       tooltip={'placement': 'bottom', 'always_visible': True},
                       updatemode='drag'),
        ]),

        html.Div(className='divider'),

        # LAS upload
        html.Div(className='sidebar-section', children=[
            html.Span('Upload LAS File', className='sidebar-label'),
            dcc.Upload(
                id='upload-las',
                children=html.Div(['Drag & drop or ', html.A('Browse', style={'color': '#3b82f6'})]),
                className='upload-area',
                accept='.las,.LAS',
            ),
            html.Div(id='upload-status', style={'fontSize': '10px', 'color': '#10b981',
                                                 'marginTop': '4px'}),
        ]),

        html.Div(className='divider'),

        # Results summary
        html.Div(className='sidebar-section', children=[
            html.Span('Analysis Results', className='sidebar-label'),
            html.Div(id='results-cards'),
        ]),

        # Facies legend
        html.Div(className='sidebar-section', children=[
            html.Span('Facies Legend', className='sidebar-label'),
            html.Div(className='facies-legend', children=[
                html.Div(className='facies-item', children=[
                    html.Div(className='facies-swatch',
                             style={'background': FACIES_COLORS[f]}),
                    html.Span(FACIES_NAMES[f]),
                ]) for f in sorted(FACIES_NAMES)
            ]),
        ]),
    ])


def _graph(gid):
    return dcc.Loading(
        dcc.Graph(id=gid, config={'displayModeBar': True},
                  style={'minHeight': '500px'}),
        color='#3b82f6',
    )


def tabs_panel():
    ts = {'padding': '6px 14px', 'fontSize': '11px'}
    tp = {'padding': '10px'}

    return dbc.Tabs(id='main-tabs', active_tab='tab-3d', children=[

        dbc.Tab(label='3-D View', tab_id='tab-3d', tab_style=ts, children=html.Div(style=tp, children=[
            view_3d_controls(),
            html.Div(className='graph-card', children=[_graph('graph-3d')]),
        ])),

        dbc.Tab(label='Well Logs', tab_id='tab-logs', tab_style=ts, children=html.Div(style=tp, children=[
            html.Div(className='graph-card', children=[_graph('graph-welllog')]),
        ])),

        dbc.Tab(label='Well Section', tab_id='tab-section', tab_style=ts, children=html.Div(style=tp, children=[
            html.Div(className='graph-card', children=[_graph('graph-section')]),
        ])),

        dbc.Tab(label='Cross Plot', tab_id='tab-xplot', tab_style=ts, children=html.Div(style=tp, children=[
            crossplot_controls(),
            html.Div(className='graph-card', children=[_graph('graph-xplot')]),
        ])),

        dbc.Tab(label='Variogram', tab_id='tab-vario', tab_style=ts, children=html.Div(style=tp, children=[
            html.Div(className='graph-card', children=[_graph('graph-vario')]),
        ])),

        dbc.Tab(label='Layer Optimisation', tab_id='tab-layeropt', tab_style=ts, children=html.Div(style=tp, children=[
            html.Div(className='graph-card', children=[_graph('graph-layeropt')]),
        ])),

        dbc.Tab(label='Log vs Upscaled', tab_id='tab-upscaled', tab_style=ts, children=html.Div(style=tp, children=[
            html.Div(className='graph-card', children=[_graph('graph-upscaled')]),
        ])),

        dbc.Tab(label='Statistics', tab_id='tab-stats', tab_style=ts, children=html.Div(style=tp, children=[
            dbc.Row([
                dbc.Col(html.Div(className='graph-card', children=[
                    html.Div('Results Summary', className='graph-header'),
                    dcc.Loading(dcc.Graph(id='graph-statstable',
                                         config={'displayModeBar': False}), color='#3b82f6'),
                ]), width=5),
                dbc.Col(html.Div(className='graph-card', children=[
                    html.Div('Marginal Gain per Layer', className='graph-header'),
                    dcc.Loading(dcc.Graph(id='graph-marginal',
                                         config={'displayModeBar': True}), color='#3b82f6'),
                ]), width=7),
            ]),
        ])),
    ])


def crossplot_controls():
    ctrl_style = {'display': 'flex', 'gap': '8px', 'padding': '8px 12px',
                  'background': '#111827', 'borderBottom': '1px solid #2d3a4f',
                  'alignItems': 'center'}
    lbl = {'fontSize': '10px', 'color': '#94a3b8', 'whiteSpace': 'nowrap', 'margin': 0}
    return html.Div(style=ctrl_style, children=[
        html.P('X:', style=lbl),
        _dropdown('xplot-x', PROPS, 'PHIE', extra_style={'width': '100px'}),
        html.P('Y:', style=lbl),
        _dropdown('xplot-y', PROPS, 'PERM', extra_style={'width': '100px'}),
        html.P('Colour:', style=lbl),
        _dropdown('xplot-color', COLOR_BY_OPT, 'FACIES', extra_style={'width': '100px'}),
        html.P('Zone:', style=lbl),
        _dropdown('xplot-zone', ALL_ZONES, 'All', extra_style={'width': '130px'}),
    ])


def view_3d_controls():
    ctrl_style = {'display': 'flex', 'gap': '8px', 'padding': '8px 12px',
                  'background': '#111827', 'borderBottom': '1px solid #2d3a4f',
                  'alignItems': 'center'}
    lbl = {'fontSize': '10px', 'color': '#94a3b8', 'margin': 0}
    return html.Div(style=ctrl_style, children=[
        html.P('Colour by:', style=lbl),
        _dropdown('3d-color-prop', PROPS, 'PHIE', extra_style={'width': '100px'}),
    ])


# ---------------------------------------------------------------------------
# Full layout
# ---------------------------------------------------------------------------
app.layout = html.Div([
    # Stores
    dcc.Store(id='store-wells', data=WELLS_INIT),
    dcc.Store(id='store-tops',  data=TOPS_INIT),
    dcc.Store(id='store-layer-results'),

    # Navbar
    html.Div(className='navbar-custom', children=[
        html.Div(style={'display': 'flex', 'flexDirection': 'column'}, children=[
            html.Span([
                html.Span('Layer', style={'color': '#3b82f6'}),
                html.Span('Optimum', style={'color': '#f59e0b'}),
                html.Span(' Pro', style={'color': '#e2e8f0', 'fontWeight': 300}),
            ], className='navbar-brand-text'),
            html.Span('Reservoir Layer Optimisation Tool', className='navbar-subtitle'),
        ]),
        html.Div(style={'marginLeft': 'auto', 'display': 'flex', 'gap': '8px',
                        'alignItems': 'center'}, children=[
            html.Span('Volve Synthetic Dataset', style={
                'fontSize': '10px', 'color': '#64748b',
                'border': '1px solid #2d3a4f', 'borderRadius': '3px',
                'padding': '2px 8px',
            }),
            html.Span(f'{len(WELLS)} Wells | {len(TOPS)} Tops',
                      style={'fontSize': '10px', 'color': '#10b981'}),
        ]),
    ]),

    # Main row
    dbc.Row(style={'margin': 0, 'padding': 0}, children=[

        # Sidebar
        dbc.Col(sidebar(), width=2,
                style={'padding': 0, 'borderRight': '1px solid #2d3a4f'}),

        # Content
        dbc.Col(width=10, style={'padding': 0}, children=[
            tabs_panel(),
        ]),
    ]),

    # Status bar
    html.Div(className='status-bar', children=[
        html.Div(className='status-item', children=[
            html.Div(className='status-dot'),
            html.Span('Ready'),
        ]),
        html.Span('Synthetic Volve-like dataset | 5 wells | 3 reservoir zones'),
        html.Span('LayerOptimum Pro v1.0 — Geoscience & Geomodelling'),
    ]),

], style={'paddingBottom': '28px'})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# Well checklist helpers
@app.callback(
    Output('well-checklist', 'value'),
    Input('btn-all-wells',  'n_clicks'),
    Input('btn-no-wells',   'n_clicks'),
    State('well-checklist', 'value'),
    prevent_initial_call=True,
)
def toggle_wells(n_all, n_none, current):
    trigger = ctx.triggered_id
    if trigger == 'btn-all-wells':
        return WELL_NAMES
    return []


# 3-D View
@app.callback(
    Output('graph-3d', 'figure'),
    Input('3d-color-prop', 'value'),
    Input('well-checklist', 'value'),
    State('store-wells', 'data'),
    State('store-tops',  'data'),
    prevent_initial_call=False,
)
def update_3d(color_prop, selected_wells, wells_json, tops_json):
    wells = _deserialize_wells(wells_json)
    tops  = pd.read_json(io.StringIO(tops_json), orient='split')
    if selected_wells:
        wells = {k: v for k, v in wells.items() if k in selected_wells}
    return figure_3d(wells, tops, color_prop or 'PHIE')


# Well Log
@app.callback(
    Output('graph-welllog', 'figure'),
    Input('well-log-selector', 'value'),
    State('store-wells', 'data'),
    State('store-tops',  'data'),
    prevent_initial_call=False,
)
def update_welllog(well_name, wells_json, tops_json):
    if not well_name:
        return _empty_fig('Select a well')
    wells = _deserialize_wells(wells_json)
    tops  = pd.read_json(io.StringIO(tops_json), orient='split')
    df    = wells.get(well_name, pd.DataFrame())
    wtops = tops[tops['WELL'] == well_name]
    return well_log_figure(df, wtops, well_name)


# Well Section
@app.callback(
    Output('graph-section', 'figure'),
    Input('well-checklist', 'value'),
    State('store-wells', 'data'),
    State('store-tops',  'data'),
    prevent_initial_call=False,
)
def update_section(selected_wells, wells_json, tops_json):
    wells = _deserialize_wells(wells_json)
    tops  = pd.read_json(io.StringIO(tops_json), orient='split')
    sel   = selected_wells or WELL_NAMES
    return well_section_figure(wells, tops, sel)


# Cross Plot
@app.callback(
    Output('graph-xplot', 'figure'),
    Input('xplot-x',     'value'),
    Input('xplot-y',     'value'),
    Input('xplot-color', 'value'),
    Input('xplot-zone',  'value'),
    State('store-wells', 'data'),
    prevent_initial_call=False,
)
def update_xplot(x_prop, y_prop, color_by, zone, wells_json):
    wells = _deserialize_wells(wells_json)
    return crossplot_figure(wells, x_prop or 'PHIE', y_prop or 'PERM',
                            color_by or 'FACIES', zone)


# Variogram
@app.callback(
    Output('graph-vario', 'figure'),
    Input('well-log-selector', 'value'),
    Input('zone-selector',     'value'),
    Input('prop-selector',     'value'),
    Input('vario-model',       'value'),
    State('store-wells', 'data'),
    State('store-tops',  'data'),
    prevent_initial_call=False,
)
def update_vario(well_name, zone_name, prop, model, wells_json, tops_json):
    if not well_name or not zone_name:
        return _empty_fig('Select well and zone')
    wells = _deserialize_wells(wells_json)
    df    = wells.get(well_name, pd.DataFrame())
    if df.empty:
        return _empty_fig('No data')
    df_zone = df[df['ZONE'] == zone_name]
    return variogram_figure(df_zone, prop or 'PHIE', well_name, zone_name, model or 'Spherical')


# Layer Optimisation + store results
@app.callback(
    Output('graph-layeropt',   'figure'),
    Output('store-layer-results', 'data'),
    Output('results-cards',    'children'),
    Input('zone-selector',     'value'),
    Input('maxlayers-slider',  'value'),
    Input('well-checklist',    'value'),
    State('store-wells', 'data'),
    prevent_initial_call=False,
)
def update_layeropt(zone_name, max_layers, selected_wells, wells_json):
    if not zone_name:
        return _empty_fig('Select a zone'), None, []

    wells = _deserialize_wells(wells_json)
    sel   = selected_wells or list(wells.keys())

    # Combine data from all selected wells for the chosen zone
    frames = []
    for wn in sel:
        df = wells.get(wn, pd.DataFrame())
        if not df.empty:
            sub = df[df['ZONE'] == zone_name]
            if not sub.empty:
                frames.append(sub)

    if not frames:
        return _empty_fig(f'No data in zone {zone_name}'), None, []

    df_combined = pd.concat(frames, ignore_index=True)
    results = optimize_layers(df_combined, max_layers=max_layers or 35)

    if results.empty:
        return _empty_fig('Analysis failed — insufficient data'), None, []

    fig = layer_analysis_figure(results, zone_name)

    # Sidebar result cards
    opt = results[results['optimal']].iloc[0] if results['optimal'].any() else results.iloc[-1]
    cards = [
        html.Div(className='result-card', children=[
            html.Div('Optimal N Layers', className='result-metric'),
            html.Div(str(int(opt['n_layers'])), className='result-value'),
            html.Div(f"VP threshold: {int(opt['vp_threshold_n'])} layers",
                     className='result-subvalue'),
        ]),
        html.Div(className='result-card', children=[
            html.Div('Variance Preservation', className='result-metric'),
            html.Div(f"{opt['variance_preservation']:.1%}", className='result-value',
                     style={'color': '#10b981', 'fontSize': '16px'}),
            html.Div(f"Lorenz: {opt['lorenz_preservation']:.1%}",
                     className='result-subvalue'),
        ]),
        html.Div(className='result-card', children=[
            html.Div('DP Coefficient (orig.)', className='result-metric'),
            html.Div(f"{opt['dp_coeff_orig']:.3f}", className='result-value',
                     style={'color': '#f59e0b', 'fontSize': '16px'}),
            html.Div(f"Lorenz coeff: {opt['lorenz_coeff_orig']:.3f}",
                     className='result-subvalue'),
        ]),
    ]
    return fig, results.to_json(orient='split'), cards


# Log vs Upscaled
@app.callback(
    Output('graph-upscaled', 'figure'),
    Input('nlayers-slider',  'value'),
    Input('zone-selector',   'value'),
    Input('well-log-selector', 'value'),
    State('store-wells', 'data'),
    prevent_initial_call=False,
)
def update_upscaled(n_layers, zone_name, well_name, wells_json):
    if not zone_name or not well_name:
        return _empty_fig('Select zone and well')
    wells   = _deserialize_wells(wells_json)
    df      = wells.get(well_name, pd.DataFrame())
    df_zone = df[df['ZONE'] == zone_name] if not df.empty else pd.DataFrame()
    return upscaled_comparison_figure(df_zone, n_layers or 10, zone_name)


# Stats table + marginal gain chart
@app.callback(
    Output('graph-statstable', 'figure'),
    Output('graph-marginal',   'figure'),
    Input('store-layer-results', 'data'),
    prevent_initial_call=False,
)
def update_stats(results_json):
    if not results_json:
        return _empty_fig('Run layer optimisation first'), _empty_fig()

    results = pd.read_json(io.StringIO(results_json), orient='split')
    tbl = summary_table_figure(results)

    # Marginal gain bar chart
    from src.figures import BG, PANEL, GRID, TEXT, TEXT2, AMBER, GREEN, BORDER
    opt_n = int(results.loc[results['optimal'], 'n_layers'].values[0]) if results['optimal'].any() else 1
    colors = [AMBER if n == opt_n else '#3b82f6' for n in results['n_layers'].values]

    import plotly.graph_objects as go
    mg_fig = go.Figure(go.Bar(
        x=results['n_layers'].values,
        y=results['marginal_gain'].values,
        marker=dict(color=colors, opacity=0.85),
        hovertemplate='N=%{x}<br>Marginal gain=%{y:.4f}<extra></extra>',
    ))
    mg_fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL, height=300,
        font=dict(color=TEXT, size=11), margin=dict(l=50, r=20, t=20, b=40),
        xaxis=dict(title='Number of Layers', gridcolor=GRID, tickfont=dict(color=TEXT2)),
        yaxis=dict(title='Marginal gain in score', gridcolor=GRID, tickfont=dict(color=TEXT2)),
        showlegend=False,
    )
    mg_fig.add_vline(x=opt_n, line=dict(color='#ffd700', width=2, dash='dash'))
    mg_fig.add_annotation(x=opt_n, y=results['marginal_gain'].max(),
                           text=f'Opt N={opt_n}', font=dict(color='#ffd700', size=11),
                           showarrow=False, yshift=8)
    return tbl, mg_fig


# LAS file upload
@app.callback(
    Output('store-wells',   'data'),
    Output('upload-status', 'children'),
    Input('upload-las', 'contents'),
    State('upload-las', 'filename'),
    State('store-wells', 'data'),
    prevent_initial_call=True,
)
def load_las(contents, filename, wells_json):
    if contents is None:
        return wells_json, ''
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        las = lasio.read(io.StringIO(decoded.decode('utf-8', errors='replace')))
        df  = las.df().reset_index()
        df.columns = [c.upper() for c in df.columns]
        if 'DEPT' in df.columns and 'DEPTH' not in df.columns:
            df = df.rename(columns={'DEPT': 'DEPTH'})
        df['ZONE']   = 'Uploaded'
        df['X']      = 0.0
        df['Y']      = 0.0
        df['TVDSS']  = df.get('DEPTH', 0)
        if 'FACIES' not in df.columns:
            df['FACIES'] = 0
        wells = _deserialize_wells(wells_json)
        well_name = filename.replace('.las', '').replace('.LAS', '')
        wells[well_name] = df
        return _serialize_wells(wells), f'Loaded: {well_name}'
    except Exception as e:
        return wells_json, f'Error: {str(e)[:60]}'


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)
