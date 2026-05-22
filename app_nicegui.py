"""LayerOptimum Pro — NiceGUI + ECharts + Plotly interface (v3.0)."""

import copy, io, csv
try:
    import lasio
except ImportError:
    lasio = None
import numpy as np
import pandas as pd
from nicegui import ui

from src.synthetic import (
    generate_synthetic_wells, generate_formation_tops,
    WELL_NAMES, RESERVOIR_ZONES, ZONES as ZONE_DEPTHS,
)
from src.analysis import (
    optimize_layers, compute_variogram, fit_variogram, compute_model_curve,
    recommend_from_variogram, heterogeneity_coarsening, compute_hcpv_proxy,
)
from src.figures import (
    well_log_figure, petrel_upscaled_figure, petrel_upscaled_allzones,
    ZONE_COLORS, FACIES_COLORS,
)
from src.echarts import (
    variogram_echart, layer_metric_echart, crossplot_echart,
    heterogeneity_echart, grid_search_heatmap, stats_rows, zone_thickness_echart,
    BLUE, GREEN, AMBER, PURPLE, GOLD, RED, CYAN, BG, PANEL, BORDER, TEXT, TEXT2,
)
from src.synthetic import FACIES_NAMES as SYN_FACIES_NAMES

# ── Data ──────────────────────────────────────────────────────────────────────
WELLS = generate_synthetic_wells()
TOPS  = generate_formation_tops(WELLS)
PROPS = ['GR', 'PHIE', 'PERM', 'SW', 'NTG', 'NPHI', 'RHOB']
ALL_ZONES = ['All'] + list(ZONE_COLORS.keys())

VARIO_PROPS = [
    ('PHIE', BLUE),
    ('PERM', AMBER),
    ('SW',   CYAN),
]
VARIO_PROP_KEYS = [p[0] for p in VARIO_PROPS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zone_df(well: str, zone: str) -> pd.DataFrame:
    df = WELLS.get(well, pd.DataFrame())
    return df[df['ZONE'] == zone].copy() if not df.empty else df


def _combined_zone(wells_sel: list, zone: str) -> pd.DataFrame:
    frames = [WELLS[w][WELLS[w]['ZONE'] == zone]
              for w in wells_sel if w in WELLS and not WELLS[w].empty]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _all_df(wells_sel: list, zone_filter: str = 'All') -> pd.DataFrame:
    frames = []
    for wn in wells_sel:
        df = WELLS.get(wn, pd.DataFrame())
        if df.empty:
            continue
        tmp = df.copy()
        tmp['WELL'] = wn
        if zone_filter != 'All':
            tmp = tmp[tmp['ZONE'] == zone_filter]
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _set_echart(chart, opts: dict):
    chart.options.clear()
    chart.options.update(copy.deepcopy(opts))
    chart.update()


def _set_plotly(chart, fig):
    chart.figure = fig
    chart.update()


def _run_all_zones(wells_sel: list, max_layers: int) -> dict:
    results = {}
    for zone in RESERVOIR_ZONES:
        df_z = _combined_zone(wells_sel, zone)
        results[zone] = optimize_layers(df_z, max_layers=max_layers) if not df_z.empty else pd.DataFrame()
    return results


def _zone_thickness(well: str, zone: str) -> float:
    df = _zone_df(well, zone)
    if df.empty:
        return 0.0
    return float(df['DEPTH'].max() - df['DEPTH'].min())


# ── App ───────────────────────────────────────────────────────────────────────

@ui.page('/')
def index():
    S = {
        'wells':        list(WELL_NAMES),
        'log_well':     WELL_NAMES[0],
        'zone':         RESERVOIR_ZONES[-1],
        'prop':         'PHIE',
        'n_layers':     10,
        'max_layers':   35,
        'vario_model':  'Spherical',
        'x_prop':       'PHIE',
        'y_prop':       'PERM',
        'color_by':     'FACIES',
        'zone_filter':  'All',
        'results':      None,
        'all_results':  {},
        'vario_params': {},
        'target_pct':   0.85,       # target preservation %
        'hetero_target': 80.0,      # target heterogeneity %
        'all_zones_up': False,      # show all zones in log vs upscaled
        'auto_refresh': False,
        'refresh_sec':  30,
    }

    charts       = {}
    labels       = {}
    vario_inputs = {}
    tops_inputs  = {}    # {well: {zone: ui.number}}
    _refresh_timer = [None]

    # ── Update functions ──────────────────────────────────────────────────────

    def upd_welllog():
        df    = WELLS.get(S['log_well'], pd.DataFrame())
        wtops = TOPS[TOPS['WELL'] == S['log_well']]
        _set_plotly(charts['welllog'], well_log_figure(df, wtops, S['log_well']))

    def upd_xplot():
        df = _all_df(S['wells'], S['zone_filter'])
        if df.empty:
            return
        _set_echart(charts['xplot'],
                    crossplot_echart(df, S['x_prop'], S['y_prop'], S['color_by']))

    def upd_vario_panel(prop: str):
        df_z = _zone_df(S['log_well'], S['zone'])
        if df_z.empty or prop not in df_z.columns:
            return
        dz   = float(df_z['DEPTH'].diff().median())
        lags, gamma, _ = compute_variogram(df_z[prop].values, dz, n_lags=30)

        params = S['vario_params'].get(prop)
        if params and params.get('manual'):
            nugget = params['nugget']
            sill   = params['sill']
            range_ = params['range_']
            h_fit  = np.linspace(0, lags[-1] * 1.1, 200)
            g_fit  = compute_model_curve(h_fit, S['vario_model'], nugget, sill, range_)
            res = {'model': S['vario_model'], 'nugget': nugget,
                   'sill': sill, 'range': range_, 'h_fit': h_fit, 'g_fit': g_fit}
        else:
            res = fit_variogram(lags, gamma, S['vario_model'])
            S['vario_params'][prop] = {
                'nugget': res['nugget'], 'sill': res['sill'],
                'range_': res['range'], 'manual': False}
            if prop in vario_inputs:
                vario_inputs[prop]['nugget'].value  = round(res['nugget'], 6)
                vario_inputs[prop]['sill'].value    = round(res['sill'],   6)
                vario_inputs[prop]['range_'].value  = round(res['range'],  2)

        _set_echart(charts[f'vario_{prop}'],
                    variogram_echart(lags, gamma, res['h_fit'], res['g_fit'], res))

        thick = _zone_thickness(S['log_well'], S['zone'])
        rec_n = recommend_from_variogram(thick, res['range'])
        if f'vario_lbl_{prop}' in labels:
            labels[f'vario_lbl_{prop}'].set_text(
                f"Zone thickness: {thick:.1f} m  |  Range: {res['range']:.1f} m  |  "
                f"Sill: {res['sill']:.5f}  |  Nugget: {res['nugget']:.5f}  →  Min layers: {rec_n}")

    def upd_thickness_chart():
        thicknesses = [_zone_thickness(wn, S['zone']) for wn in WELL_NAMES]
        _set_echart(charts['vario_thick'],
                    zone_thickness_echart(list(WELL_NAMES), thicknesses, S['zone']))
        avg_t = float(np.mean(thicknesses)) if thicknesses else 0.0
        if 'vario_lbl_thick' in labels:
            labels['vario_lbl_thick'].set_text(
                f"Zone: {S['zone'].replace('_',' ')}  |  "
                f"Avg: {avg_t:.1f} m  |  "
                f"Min: {min(thicknesses):.1f} m  |  Max: {max(thicknesses):.1f} m")

    def upd_vario():
        for prop in VARIO_PROP_KEYS:
            upd_vario_panel(prop)
        upd_thickness_chart()

    def upd_layer():
        df_z = _combined_zone(S['wells'], S['zone'])
        if df_z.empty:
            return
        results = optimize_layers(df_z, max_layers=S['max_layers'])
        S['results'] = results
        if results.empty:
            return

        opt_n = int(results.loc[results['optimal'], 'n_layers'].iloc[0]) \
                if results['optimal'].any() else 1
        nl    = results['n_layers'].values.tolist()

        metrics = [
            ('vp',  'variance_preservation', 'Variance Preservation', BLUE,   0.85),
            ('lp',  'lorenz_preservation',   'Lorenz Preservation',   GREEN,  0.85),
            ('dp',  'dp_preservation',       'DP Preservation',       AMBER,  0.80),
            ('fc',  'facies_coverage',       'Facies Coverage',       PURPLE, 1.00),
        ]
        for key, col, lbl, color, thr in metrics:
            _set_echart(charts[key],
                        layer_metric_echart(nl, results[col].values.tolist(),
                                            lbl, opt_n, color, thr,
                                            target_pct=S['target_pct']))

        # Heterogeneity coarsening (GeoConvention 2010 methodology)
        het_res = heterogeneity_coarsening(df_z)
        if not het_res.empty:
            _set_echart(charts['hetero'],
                        heterogeneity_echart(het_res, S['zone'],
                                             target_pct=S['hetero_target']))
            het_opt = int(het_res.loc[het_res['inflection'], 'n_layers'].iloc[0]) \
                      if het_res['inflection'].any() else opt_n
            if 'hetero_n' in labels:
                labels['hetero_n'].set_text(f"Elbow N = {het_opt}")

        # HCPV proxy
        hcpv = compute_hcpv_proxy(df_z)
        zone_thick = _zone_thickness(S['wells'][0] if S['wells'] else WELL_NAMES[0], S['zone'])

        opt = results[results['optimal']].iloc[0] if results['optimal'].any() \
              else results.iloc[-1]
        labels['opt_n'].set_text(str(int(opt['n_layers'])))
        labels['score'].set_text(f"Score: {opt['combined_score']:.3f}")
        labels['vp'].set_text(f"{opt['variance_preservation']:.1%}")
        labels['lp'].set_text(f"{opt['lorenz_preservation']:.1%}")
        labels['dp'].set_text(f"{opt['dp_preservation']:.1%}")
        labels['fc'].set_text(f"{opt['facies_coverage']:.1%}")
        labels['lc_orig'].set_text(f"{opt['lorenz_coeff_orig']:.3f}")
        labels['dp_orig'].set_text(f"{opt['dp_coeff_orig']:.3f}")
        if 'hcpv_lbl' in labels:
            labels['hcpv_lbl'].set_text(f"HCPV proxy: {hcpv.sum():.2f} m")
        if 'thick_lbl' in labels:
            labels['thick_lbl'].set_text(f"Zone thickness: {zone_thick:.1f} m")

    def upd_upscaled():
        wtops = TOPS[TOPS['WELL'] == S['log_well']]
        if S['all_zones_up']:
            _set_plotly(charts['upscaled'],
                        petrel_upscaled_allzones(
                            WELLS, RESERVOIR_ZONES,
                            {z: S['n_layers'] for z in RESERVOIR_ZONES},
                            S['log_well'], tops_df=wtops))
        else:
            df_z = _zone_df(S['log_well'], S['zone'])
            _set_plotly(charts['upscaled'],
                        petrel_upscaled_figure(df_z, S['n_layers'], S['zone'],
                                               tops_df=wtops))

    def upd_stats():
        all_res = _run_all_zones(S['wells'], S['max_layers'])
        S['all_results'] = all_res

        zone_rows = []
        for zname, df_res in all_res.items():
            if df_res.empty:
                zone_rows.append({'zone': zname.replace('_', ' '), 'opt_n': '—',
                                  'vp': '—', 'lp': '—', 'dp': '—', 'fc': '—',
                                  'score': '—', 'thick': '—', 'hcpv': '—'})
                continue
            opt = df_res[df_res['optimal']].iloc[0] if df_res['optimal'].any() \
                  else df_res.iloc[-1]
            df_z  = _combined_zone(S['wells'], zname)
            thick = float(df_z['DEPTH'].max() - df_z['DEPTH'].min()) if not df_z.empty else 0
            hcpv  = float(compute_hcpv_proxy(df_z).sum()) if not df_z.empty else 0
            zone_rows.append({
                'zone':  zname.replace('_', ' '),
                'opt_n': str(int(opt['n_layers'])),
                'vp':    f"{opt['variance_preservation']:.1%}",
                'lp':    f"{opt['lorenz_preservation']:.1%}",
                'dp':    f"{opt['dp_preservation']:.1%}",
                'fc':    f"{opt['facies_coverage']:.1%}",
                'score': f"{opt['combined_score']:.3f}",
                'thick': f"{thick:.1f} m",
                'hcpv':  f"{hcpv:.2f} m",
            })
        charts['zone_table'].rows = zone_rows
        charts['zone_table'].update()

        if S['results'] is not None and not S['results'].empty:
            charts['stats_table'].rows = stats_rows(S['results'])
            charts['stats_table'].update()

        valid = {z: r for z, r in all_res.items() if not r.empty}
        if valid:
            _set_echart(charts['heatmap'], grid_search_heatmap(valid))

    def upd_tops_inputs():
        """Refresh tops-edit widgets from current TOPS dataframe."""
        for wn, zone_map in tops_inputs.items():
            wtops = TOPS[TOPS['WELL'] == wn]
            for zname, inp in zone_map.items():
                row = wtops[wtops['ZONE'] == zname]
                if not row.empty:
                    inp.value = float(row['TOP'].iloc[0])

    def upd_all():
        upd_welllog()
        upd_xplot()
        upd_vario()
        upd_layer()
        upd_upscaled()
        upd_stats()

    # ── Auto refresh ──────────────────────────────────────────────────────────

    def _restart_timer():
        if _refresh_timer[0] is not None:
            _refresh_timer[0].cancel()
            _refresh_timer[0] = None
        if S['auto_refresh']:
            _refresh_timer[0] = ui.timer(S['refresh_sec'], upd_all)

    # ── CSS ──────────────────────────────────────────────────────────────────
    ui.add_css('''
        body { background:#0a0e1a !important;
               font-family:"Segoe UI",Inter,sans-serif !important; }
        .nicegui-content { background:#0a0e1a !important; }
        .q-header { background:linear-gradient(90deg,#0d1829,#0f2040,#0d1829) !important;
                    border-bottom:2px solid #3b82f6 !important;
                    box-shadow:0 2px 14px rgba(59,130,246,.22) !important; }
        .q-drawer { background:#111827 !important;
                    border-right:1px solid #2d3a4f !important; }
        .q-page  { background:#0a0e1a !important; }
        .q-footer { background:#111827 !important;
                    border-top:1px solid #2d3a4f !important; }
        .q-tabs  { background:#161e2e !important;
                   border-bottom:1px solid #2d3a4f !important; }
        .q-tab   { color:#94a3b8 !important; font-size:11px !important;
                   min-height:38px !important; }
        .q-tab--active { color:#3b82f6 !important;
                         border-top:2px solid #3b82f6; font-weight:600; }
        .q-tab-panel { padding:10px !important; background:#0a0e1a !important; }
        .q-field__control  { background:#1a2332 !important;
                              border:1px solid #2d3a4f !important;
                              border-radius:5px !important; }
        .q-field__native, .q-field__label, .q-item__label
                            { color:#e2e8f0 !important; font-size:11px !important; }
        .q-menu  { background:#1a2332 !important; border:1px solid #2d3a4f !important; }
        .q-item:hover { background:#2d3a4f !important; }
        .q-slider__track     { background:#2d3a4f !important; }
        .q-slider__selection { background:#3b82f6 !important; }
        .q-slider__thumb     { background:#3b82f6 !important; border-color:#3b82f6 !important; }
        ::-webkit-scrollbar { width:5px; height:5px; }
        ::-webkit-scrollbar-track { background:#0a0e1a; }
        ::-webkit-scrollbar-thumb { background:#2d3a4f; border-radius:3px; }
        .q-table thead tr th { background:#161e2e !important; color:#f59e0b !important;
                                font-size:11px !important; border-color:#2d3a4f !important; }
        .q-table tbody tr td { color:#e2e8f0 !important; font-size:11px !important;
                                border-color:#1e2d42 !important; }
        .q-table tbody tr:hover td { background:#1a2332 !important; }
        .geo-card { background:#111827 !important; border:1px solid #2d3a4f !important;
                    border-radius:6px !important; overflow:hidden; }
        .geo-header { background:#161e2e !important; border-bottom:1px solid #2d3a4f !important;
                      padding:6px 14px !important; font-size:10px !important;
                      font-weight:700; letter-spacing:2px; text-transform:uppercase;
                      color:#f59e0b !important; }
        .s-label { font-size:9px !important; font-weight:700 !important;
                   letter-spacing:2px !important; text-transform:uppercase !important;
                   color:#f59e0b !important; display:block; padding:2px 0 4px 0;
                   border-bottom:1px solid #2d3a4f; margin-bottom:6px; }
        .vario-param { background:#0f1c2e !important; border:1px solid #1e3a5f !important;
                       border-radius:4px; padding:4px 8px; }
        .info-strip { background:#0f1c2e; border:1px solid #1e3a5f; border-radius:4px;
                      padding:4px 10px; font-size:10px; color:#06b6d4; }
        .q-expansion-item__header { background:#161e2e !important;
                                     border:1px solid #2d3a4f !important;
                                     border-radius:4px !important; }
    ''')

    # ── Header ────────────────────────────────────────────────────────────────
    with ui.header().classes('items-center px-4 h-13 min-h-[52px]'):
        ui.html(
            '<span style="font-size:18px;font-weight:700;letter-spacing:1px">'
            '<span style="color:#3b82f6">Layer</span>'
            '<span style="color:#f59e0b">Optimum</span>'
            '<span style="color:#e2e8f0;font-weight:300"> Pro</span>'
            '</span>'
            '<span style="font-size:10px;color:#64748b;letter-spacing:2px;'
            'text-transform:uppercase;margin-left:12px">Reservoir Layer Optimisation</span>'
        )
        ui.space()
        ui.badge('GeoConvention 2010 Methodology', color='teal').classes('text-[9px]')
        ui.badge('Volve Synthetic Dataset', color='blue-grey').classes('text-[10px] ml-1')
        ui.label(f'{len(WELLS)} Wells  |  {len(TOPS)} Tops').classes(
            'text-[11px] text-green-400 ml-3')
        ui.button(icon='menu', on_click=lambda: drawer.toggle()).props(
            'flat round dense').classes('text-white ml-2')

    # ── Left drawer ───────────────────────────────────────────────────────────
    with ui.left_drawer(fixed=True, bordered=False).style(
            'background:#111827;border-right:1px solid #2d3a4f;width:230px') as drawer:
        with ui.scroll_area().classes('h-full').style('padding:10px 8px'):

            # Wells
            ui.html('<span class="s-label">Wells</span>')
            well_checks = {}
            for wn in WELL_NAMES:
                chk = ui.checkbox(wn, value=True).classes('text-[11px] text-gray-400')
                chk.props('dense color=blue')
                def _on_well(e, w=wn):
                    if e.value:
                        if w not in S['wells']:
                            S['wells'].append(w)
                    elif w in S['wells']:
                        S['wells'].remove(w)
                chk.on_value_change(_on_well)
                well_checks[wn] = chk
            with ui.row().classes('gap-1 mt-1 w-full'):
                def _all_w():
                    S['wells'] = list(WELL_NAMES)
                    for c in well_checks.values(): c.value = True
                def _none_w():
                    S['wells'] = []
                    for c in well_checks.values(): c.value = False
                ui.button('All', on_click=_all_w).props('flat dense size=xs').classes(
                    'flex-1 text-gray-400').style('border:1px solid #2d3a4f')
                ui.button('None', on_click=_none_w).props('flat dense size=xs').classes(
                    'flex-1 text-gray-400').style('border:1px solid #2d3a4f')

            ui.separator().style('background:#2d3a4f;margin:10px 0')

            # Log display well
            ui.html('<span class="s-label">Log Display Well</span>')
            def _on_log_well(e):
                S['log_well'] = e.value
                upd_welllog()
                upd_tops_inputs()
            ui.select(options=WELL_NAMES, value=S['log_well'],
                      on_change=_on_log_well).classes('w-full text-[11px]').props('dense')

            ui.separator().style('background:#2d3a4f;margin:10px 0')

            # Analysis zone
            ui.html('<span class="s-label">Analysis Zone</span>')
            def _on_zone(e):
                S['zone'] = e.value
                upd_vario(); upd_layer(); upd_upscaled()
            ui.select(options=RESERVOIR_ZONES, value=S['zone'],
                      on_change=_on_zone).classes('w-full text-[11px]').props('dense')

            ui.separator().style('background:#2d3a4f;margin:10px 0')

            # N layers slider
            ui.html('<span class="s-label">N Layers (upscaling)</span>')
            n_lbl = ui.label(f'N = {S["n_layers"]}').classes(
                'text-[11px] text-blue-400 text-center w-full')
            def _on_n(e):
                S['n_layers'] = int(e.value)
                n_lbl.set_text(f'N = {S["n_layers"]}')
                upd_upscaled()
            ui.slider(min=1, max=40, value=S['n_layers'],
                      on_change=_on_n).classes('w-full').props('dense')

            # Target preservation %
            ui.html('<span class="s-label mt-2">Target Preservation %</span>')
            tgt_lbl = ui.label(f'{S["target_pct"]:.0%}').classes(
                'text-[11px] text-red-400 text-center w-full')
            def _on_tgt(e):
                S['target_pct'] = float(e.value) / 100.0
                tgt_lbl.set_text(f'{S["target_pct"]:.0%}')
                upd_layer()
            ui.slider(min=50, max=99, step=1, value=int(S['target_pct'] * 100),
                      on_change=_on_tgt).classes('w-full').props('dense color=red')

            # Heterogeneity target %
            ui.html('<span class="s-label mt-2">Heterogeneity Target %</span>')
            htgt_lbl = ui.label(f'{S["hetero_target"]:.0f}%').classes(
                'text-[11px] text-cyan-400 text-center w-full')
            def _on_htgt(e):
                S['hetero_target'] = float(e.value)
                htgt_lbl.set_text(f'{S["hetero_target"]:.0f}%')
                upd_layer()
            ui.slider(min=50, max=99, step=1, value=int(S['hetero_target']),
                      on_change=_on_htgt).classes('w-full').props('dense color=cyan')

            # Variogram model
            ui.html('<span class="s-label mt-2">Variogram Model</span>')
            def _on_vmod(e):
                S['vario_model'] = e.value
                for p in VARIO_PROP_KEYS:
                    if p in S['vario_params']:
                        S['vario_params'][p]['manual'] = False
                upd_vario()
            ui.select(options=['Spherical', 'Exponential', 'Gaussian'],
                      value=S['vario_model'],
                      on_change=_on_vmod).classes('w-full text-[11px]').props('dense')

            # Max layers
            ui.html('<span class="s-label mt-2">Max Layers (analysis)</span>')
            ml_lbl = ui.label(f'Max = {S["max_layers"]}').classes(
                'text-[11px] text-amber-400 text-center w-full')
            def _on_ml(e):
                S['max_layers'] = int(e.value)
                ml_lbl.set_text(f'Max = {S["max_layers"]}')
                upd_layer()
            ui.slider(min=5, max=60, step=5, value=S['max_layers'],
                      on_change=_on_ml).classes('w-full').props('dense')

            ui.separator().style('background:#2d3a4f;margin:10px 0')

            # Auto refresh
            ui.html('<span class="s-label">Auto Refresh</span>')
            with ui.row().classes('items-center gap-2 w-full'):
                def _on_auto(e):
                    S['auto_refresh'] = e.value
                    _restart_timer()
                ui.switch(value=False, on_change=_on_auto).props('color=blue dense')
                ui.label('Every').classes('text-[10px] text-gray-400')
                def _on_rsec(e):
                    S['refresh_sec'] = max(5, int(e.value or 30))
                    _restart_timer()
                ui.number(value=30, min=5, max=300, step=5,
                          on_change=_on_rsec).classes('text-[10px]').style(
                    'width:60px').props('dense outlined suffix=s')

            ui.separator().style('background:#2d3a4f;margin:10px 0')

            # Refresh button
            ui.button('⟳  Refresh All', on_click=upd_all).classes('w-full text-[11px]').style(
                'border:1px solid #3b82f6;color:#3b82f6;background:transparent')

            ui.separator().style('background:#2d3a4f;margin:10px 0')

            # Upload LAS
            ui.html('<span class="s-label">Upload LAS</span>')
            def _on_upload_las(e):
                try:
                    if lasio is None:
                        las_lbl.set_text('Error: lasio not installed'); return
                    las = lasio.read(io.StringIO(e.content.read().decode('utf-8', errors='replace')))
                    df  = las.df().reset_index()
                    df.columns = [c.upper() for c in df.columns]
                    if 'DEPT' in df.columns and 'DEPTH' not in df.columns:
                        df = df.rename(columns={'DEPT': 'DEPTH'})
                    df['ZONE'] = 'Uploaded'; df['X'] = 0.0; df['Y'] = 0.0
                    df['TVDSS'] = df.get('DEPTH', pd.Series(dtype=float))
                    if 'FACIES' not in df.columns: df['FACIES'] = 0
                    if 'NTG' not in df.columns:    df['NTG'] = 1.0
                    if 'SW' not in df.columns:     df['SW']  = 0.3
                    wn = e.name.replace('.las','').replace('.LAS','')
                    WELLS[wn] = df
                    TOPS.loc[len(TOPS)] = {'WELL': wn, 'ZONE': 'Uploaded',
                                           'TOP': df['DEPTH'].min(),
                                           'BASE': df['DEPTH'].max(),
                                           'THICKNESS': 0, 'X': 0, 'Y': 0}
                    las_lbl.set_text(f'✓ Loaded: {wn}')
                except Exception as ex:
                    las_lbl.set_text(f'Error: {str(ex)[:50]}')
            ui.upload(on_upload=_on_upload_las, label='Drop LAS here').classes(
                'w-full text-[10px]').props('accept=".las,.LAS" flat dense')
            las_lbl = ui.label('').classes('text-[10px] text-green-400 mt-1')

            # Upload Formation Tops CSV
            ui.html('<span class="s-label mt-2">Upload Formation Tops (CSV)</span>')
            def _on_upload_tops(e):
                try:
                    content = e.content.read().decode('utf-8', errors='replace')
                    reader  = csv.DictReader(io.StringIO(content))
                    count   = 0
                    for row in reader:
                        well = row.get('WELL', '').strip()
                        zone = row.get('ZONE', '').strip()
                        top  = float(row.get('TOP', 0))
                        mask = (TOPS['WELL'] == well) & (TOPS['ZONE'] == zone)
                        if mask.any():
                            TOPS.loc[mask, 'TOP'] = top
                        else:
                            TOPS.loc[len(TOPS)] = {'WELL': well, 'ZONE': zone,
                                                   'TOP': top, 'BASE': top + 10,
                                                   'THICKNESS': 0, 'X': 0, 'Y': 0}
                        count += 1
                    tops_lbl.set_text(f'✓ {count} tops loaded')
                    upd_tops_inputs()
                    upd_welllog()
                except Exception as ex:
                    tops_lbl.set_text(f'Error: {str(ex)[:50]}')
            ui.upload(on_upload=_on_upload_tops, label='WELL,ZONE,TOP CSV').classes(
                'w-full text-[10px]').props('accept=".csv,.CSV" flat dense')
            tops_lbl = ui.label('').classes('text-[10px] text-amber-400 mt-1')

            ui.separator().style('background:#2d3a4f;margin:10px 0')

            # Results panel
            ui.html('<span class="s-label">Analysis Results</span>')
            with ui.card().classes('w-full mb-2').style(
                    'background:#1a2332;border:1px solid #2d3a4f;padding:10px'):
                ui.label('Optimal N Layers').classes('text-[10px] text-gray-400')
                labels['opt_n'] = ui.label('—').classes(
                    'text-[20px] font-bold text-[#ffd700] leading-tight')
                labels['score'] = ui.label('Score: —').classes('text-[10px] text-green-400')
                labels['hcpv_lbl']  = ui.label('HCPV proxy: —').classes('text-[10px] text-cyan-400')
                labels['thick_lbl'] = ui.label('Zone thickness: —').classes('text-[10px] text-gray-400')
                labels['hetero_n']  = ui.label('Elbow N = —').classes('text-[10px] text-cyan-300')

            with ui.card().classes('w-full mb-2').style(
                    'background:#1a2332;border:1px solid #2d3a4f;padding:10px'):
                with ui.grid(columns=2).classes('w-full gap-1'):
                    for key, lbl_txt, clr in [
                        ('vp', 'Variance Pres.', 'text-green-400'),
                        ('lp', 'Lorenz Pres.',   'text-green-400'),
                        ('dp', 'DP Pres.',        'text-amber-400'),
                        ('fc', 'Facies Cov.',     'text-purple-400'),
                    ]:
                        ui.label(lbl_txt).classes('text-[9px] text-gray-500')
                        labels[key] = ui.label('—').classes(f'text-[11px] font-bold {clr}')

            with ui.card().classes('w-full').style(
                    'background:#1a2332;border:1px solid #2d3a4f;padding:10px'):
                with ui.grid(columns=2).classes('w-full gap-1'):
                    ui.label('LC orig').classes('text-[9px] text-gray-500')
                    labels['lc_orig'] = ui.label('—').classes('text-[11px] text-amber-400')
                    ui.label('DP orig').classes('text-[9px] text-gray-500')
                    labels['dp_orig'] = ui.label('—').classes('text-[11px] text-amber-400')

            ui.separator().style('background:#2d3a4f;margin:10px 0')

            # Facies legend
            ui.html('<span class="s-label">Facies Legend</span>')
            for fc_code, fc_name in sorted(SYN_FACIES_NAMES.items()):
                with ui.row().classes('items-center gap-2 mb-0.5'):
                    ui.html(f'<div style="width:11px;height:11px;background:'
                            f'{FACIES_COLORS[fc_code]};border-radius:2px;flex-shrink:0"></div>')
                    ui.label(fc_name).classes('text-[10px] text-gray-400')

    # ── Main tabs ─────────────────────────────────────────────────────────────
    with ui.column().classes('w-full').style('background:#0a0e1a'):

        with ui.tabs().classes('w-full').style(
                'background:#161e2e;border-bottom:1px solid #2d3a4f') as tabs:
            for tab_id, lbl, icon in [
                ('logs',    'Well Logs',          'show_chart'),
                ('xplot',   'Cross Plot',         'scatter_plot'),
                ('vario',   'Variogram',          'timeline'),
                ('layer',   'Layer Optimisation', 'tune'),
                ('up',      'Log vs Upscaled',    'compare'),
                ('stats',   'Statistics',         'bar_chart'),
            ]:
                ui.tab(tab_id, label=lbl, icon=icon)

        with ui.tab_panels(tabs, value='logs').classes('w-full').style(
                'background:#0a0e1a;padding:10px'):

            # ── Well Logs ─────────────────────────────────────────────────────
            with ui.tab_panel('logs'):
                with ui.element('div').classes('geo-card'):
                    df0 = WELLS[WELL_NAMES[0]]
                    t0  = TOPS[TOPS['WELL'] == WELL_NAMES[0]]
                    charts['welllog'] = ui.plotly(
                        well_log_figure(df0, t0, WELL_NAMES[0])
                    ).classes('w-full').style('height:720px')

                # ── Formation Tops Editor ─────────────────────────────────────
                with ui.expansion('Edit Formation Tops', icon='edit').classes(
                        'w-full mt-2').style(
                        'background:#111827;border:1px solid #2d3a4f;border-radius:6px'):
                    ui.label('Edit top depths (m) for the selected log display well. '
                             'Click Apply to update the well log.').classes(
                        'text-[10px] text-gray-400 px-2 pt-1')

                    for wn in WELL_NAMES:
                        tops_inputs[wn] = {}
                        with ui.expansion(wn, icon='oil_barrel').classes('w-full').style(
                                'background:#0f1c2e;margin:4px 0;border-radius:4px'):
                            with ui.grid(columns=3).classes('w-full gap-2 p-2'):
                                wtops = TOPS[TOPS['WELL'] == wn]
                                for zname in list(ZONE_COLORS.keys()):
                                    row = wtops[wtops['ZONE'] == zname]
                                    init_v = float(row['TOP'].iloc[0]) if not row.empty else 0.0
                                    zclr   = ZONE_COLORS.get(zname, '#888')
                                    ui.html(f'<span style="color:{zclr};font-size:10px;'
                                            f'font-weight:600">{zname.replace("_"," ")}</span>')
                                    inp = ui.number(value=init_v, format='%.1f',
                                                    step=0.5).classes('text-[10px]').style(
                                        'width:90px').props('dense outlined suffix=m')
                                    tops_inputs[wn][zname] = inp
                                    ui.label('')  # spacer

                    def _apply_tops():
                        for wn, zone_map in tops_inputs.items():
                            for zname, inp in zone_map.items():
                                new_top = float(inp.value)
                                mask = (TOPS['WELL'] == wn) & (TOPS['ZONE'] == zname)
                                if mask.any():
                                    TOPS.loc[mask, 'TOP'] = new_top
                                else:
                                    TOPS.loc[len(TOPS)] = {
                                        'WELL': wn, 'ZONE': zname, 'TOP': new_top,
                                        'BASE': new_top + 10, 'THICKNESS': 0, 'X': 0, 'Y': 0}
                        upd_welllog()
                        ui.notify('Formation tops updated', color='positive')

                    ui.button('Apply Formation Tops', on_click=_apply_tops).classes(
                        'mt-2 ml-2 mb-2 text-[11px]').style(
                        'border:1px solid #10b981;color:#10b981;background:transparent')

            # ── Cross Plot ────────────────────────────────────────────────────
            with ui.tab_panel('xplot'):
                with ui.row().classes('items-center gap-3 mb-2').style(
                        'background:#111827;border:1px solid #2d3a4f;'
                        'border-radius:5px;padding:6px 12px'):
                    ui.label('X:').classes('text-[10px] text-gray-400')
                    xp_x = ui.select(options=PROPS, value='PHIE').classes(
                        'w-20 text-[11px]').props('dense')
                    ui.label('Y:').classes('text-[10px] text-gray-400')
                    xp_y = ui.select(options=PROPS, value='PERM').classes(
                        'w-20 text-[11px]').props('dense')
                    ui.label('Colour:').classes('text-[10px] text-gray-400')
                    xp_col = ui.select(options=['FACIES','ZONE'], value='FACIES').classes(
                        'w-20 text-[11px]').props('dense')
                    ui.label('Zone:').classes('text-[10px] text-gray-400')
                    xp_zf = ui.select(options=ALL_ZONES, value='All').classes(
                        'w-28 text-[11px]').props('dense')
                    def _on_xp(*_):
                        S['x_prop']      = xp_x.value
                        S['y_prop']      = xp_y.value
                        S['color_by']    = xp_col.value
                        S['zone_filter'] = xp_zf.value
                        upd_xplot()
                    for w in [xp_x, xp_y, xp_col, xp_zf]:
                        w.on_value_change(_on_xp)
                with ui.element('div').classes('geo-card'):
                    charts['xplot'] = ui.echart(
                        {'backgroundColor': BG, 'series': []}
                    ).classes('w-full').style('height:580px')

            # ── Variogram ─────────────────────────────────────────────────────
            with ui.tab_panel('vario'):
                with ui.grid(columns=2).classes('w-full gap-3'):
                    for prop, prop_color in VARIO_PROPS:
                        df_zi = WELLS[WELL_NAMES[0]]
                        df_zi = df_zi[df_zi['ZONE'] == RESERVOIR_ZONES[-1]]
                        dz_i  = float(df_zi['DEPTH'].diff().median())
                        l0, g0, _ = compute_variogram(df_zi[prop].values, dz_i, 30)
                        r0 = fit_variogram(l0, g0, 'Spherical')

                        with ui.element('div').classes('geo-card'):
                            ui.html(f'<div class="geo-header">{prop} Variogram</div>')
                            charts[f'vario_{prop}'] = ui.echart(
                                variogram_echart(l0, g0, r0['h_fit'], r0['g_fit'], r0)
                            ).classes('w-full').style('height:280px')

                            with ui.row().classes('items-center gap-1 px-2 py-1 vario-param flex-wrap'):
                                ui.label('Nugget').classes('text-[9px] text-gray-500')
                                inp_nug = ui.number(value=round(r0['nugget'], 6),
                                    format='%.6f', step=0.0001).classes('text-[10px]').style(
                                    'width:82px').props('dense outlined')
                                ui.label('Sill').classes('text-[9px] text-gray-500 ml-1')
                                inp_sil = ui.number(value=round(r0['sill'], 6),
                                    format='%.6f', step=0.0001).classes('text-[10px]').style(
                                    'width:82px').props('dense outlined')
                                ui.label('Range').classes('text-[9px] text-gray-500 ml-1')
                                inp_rng = ui.number(value=round(r0['range'], 2),
                                    format='%.2f', step=0.5).classes('text-[10px]').style(
                                    'width:65px').props('dense outlined suffix=m')

                                def _make_apply(p, nug_in, sil_in, rng_in):
                                    def _apply():
                                        S['vario_params'][p] = {
                                            'nugget': float(nug_in.value),
                                            'sill':   float(sil_in.value),
                                            'range_': float(rng_in.value),
                                            'manual': True,
                                        }
                                        upd_vario_panel(p)
                                    return _apply

                                def _make_autofit(p):
                                    def _autofit():
                                        if p in S['vario_params']:
                                            S['vario_params'][p]['manual'] = False
                                        upd_vario_panel(p)
                                    return _autofit

                                ui.button('Apply', on_click=_make_apply(
                                    prop, inp_nug, inp_sil, inp_rng)
                                ).props('flat dense size=xs').classes(
                                    'ml-1 text-blue-400').style(
                                    'border:1px solid #3b82f6;font-size:10px')
                                ui.button('Auto', on_click=_make_autofit(prop)
                                ).props('flat dense size=xs').classes(
                                    'text-amber-400').style(
                                    'border:1px solid #f59e0b;font-size:10px')

                            labels[f'vario_lbl_{prop}'] = ui.label('').classes(
                                'text-[10px] text-cyan-400 px-2 pb-1')

                            vario_inputs[prop] = {
                                'nugget': inp_nug, 'sill': inp_sil, 'range_': inp_rng}
                            S['vario_params'][prop] = {
                                'nugget': r0['nugget'], 'sill': r0['sill'],
                                'range_': r0['range'], 'manual': False}

                    # ── Zone Thickness per Well (4th panel — replaces NTG) ─────
                    with ui.element('div').classes('geo-card'):
                        ui.html('<div class="geo-header">Zone Thickness per Well</div>')
                        charts['vario_thick'] = ui.echart(
                            {'backgroundColor': BG, 'series': []}
                        ).classes('w-full').style('height:280px')
                        labels['vario_lbl_thick'] = ui.label('').classes(
                            'text-[10px] text-cyan-400 px-2 pb-1')

            # ── Layer Optimisation ────────────────────────────────────────────
            with ui.tab_panel('layer'):
                # Target % info strip
                with ui.row().classes('items-center gap-3 mb-2 w-full info-strip'):
                    ui.html('<span style="color:#f59e0b;font-weight:700;'
                            'font-size:10px">TARGET %</span>')
                    ui.label('Red dashed line = target preservation threshold. '
                             'Red diamond = first N that meets it. '
                             'Gold pin = elbow-optimal N.').classes('text-[10px] text-gray-400')

                _empty_opts = {
                    'backgroundColor': BG,
                    'xAxis': {'type': 'value'}, 'yAxis': {'type': 'value'},
                    'series': [],
                }
                with ui.grid(columns=2).classes('w-full gap-2'):
                    for key, lbl_txt in [
                        ('vp', 'Variance Preservation'),
                        ('lp', 'Lorenz Preservation'),
                        ('dp', 'DP Preservation'),
                        ('fc', 'Facies Coverage'),
                    ]:
                        with ui.element('div').classes('geo-card'):
                            ui.html(f'<div class="geo-header">{lbl_txt}</div>')
                            charts[key] = ui.echart(
                                copy.deepcopy(_empty_opts)
                            ).classes('w-full').style('height:280px')

                # Heterogeneity vs Layers (GeoConvention 2010)
                with ui.element('div').classes('geo-card mt-2'):
                    ui.html('<div class="geo-header">Heterogeneity vs Model Layers '
                            '(GeoConvention 2010 — greedy merge algorithm)</div>')
                    with ui.row().classes('items-center gap-2 px-3 py-1').style(
                            'background:#0f1c2e;border-bottom:1px solid #1e3a5f'):
                        ui.label('P = K/φ (flow velocity proxy)  |  '
                                 'H = Σ n·(P−P̄)²  |  '
                                 'Merge pair with min ΔW = (n₁n₂/(n₁+n₂))(P₁−P₂)²'
                                 ).classes('text-[9px] text-gray-500')
                        ui.space()
                        labels['hetero_n'] = ui.label('Elbow N = —').classes(
                            'text-[11px] font-bold text-cyan-300')
                    charts['hetero'] = ui.echart(
                        copy.deepcopy(_empty_opts)
                    ).classes('w-full').style('height:300px')

            # ── Log vs Upscaled ───────────────────────────────────────────────
            with ui.tab_panel('up'):
                with ui.row().classes('items-center gap-3 mb-2').style(
                        'background:#111827;border:1px solid #2d3a4f;'
                        'border-radius:5px;padding:6px 12px'):
                    ui.label('Display:').classes('text-[10px] text-gray-400')
                    def _on_allzones(e):
                        S['all_zones_up'] = e.value
                        upd_upscaled()
                    ui.switch('All reservoir zones', value=False,
                              on_change=_on_allzones).props('color=blue dense').classes(
                        'text-[11px] text-gray-300')

                with ui.element('div').classes('geo-card'):
                    df_z0   = _zone_df(WELL_NAMES[0], RESERVOIR_ZONES[-1])
                    tops_z0 = TOPS[TOPS['WELL'] == WELL_NAMES[0]]
                    charts['upscaled'] = ui.plotly(
                        petrel_upscaled_figure(df_z0, 10, RESERVOIR_ZONES[-1],
                                               tops_df=tops_z0)
                    ).classes('w-full').style('height:720px')

            # ── Statistics ────────────────────────────────────────────────────
            with ui.tab_panel('stats'):
                # Grid search heatmap (large, full width)
                with ui.element('div').classes('geo-card mb-3'):
                    ui.html('<div class="geo-header">Grid Search — Combined Score '
                            'Heatmap (all zones × all N)</div>')
                    charts['heatmap'] = ui.echart({
                        'backgroundColor': BG, 'series': []
                    }).classes('w-full').style('height:480px')

                # Zone-by-zone table
                with ui.element('div').classes('geo-card mb-3'):
                    ui.html('<div class="geo-header">Optimal Layers per Zone</div>')
                    zone_cols = [
                        {'name':'zone', 'label':'Zone',       'field':'zone',  'align':'left'},
                        {'name':'opt_n','label':'Opt N',      'field':'opt_n', 'align':'center'},
                        {'name':'vp',   'label':'Var Pres',   'field':'vp',    'align':'center'},
                        {'name':'lp',   'label':'Lorenz',     'field':'lp',    'align':'center'},
                        {'name':'dp',   'label':'DP Pres',    'field':'dp',    'align':'center'},
                        {'name':'fc',   'label':'Facies Cov', 'field':'fc',    'align':'center'},
                        {'name':'score','label':'Score',      'field':'score', 'align':'center'},
                        {'name':'thick','label':'Thickness',  'field':'thick', 'align':'center'},
                        {'name':'hcpv', 'label':'HCPV proxy', 'field':'hcpv',  'align':'center'},
                    ]
                    charts['zone_table'] = ui.table(
                        columns=zone_cols, rows=[], row_key='zone'
                    ).classes('w-full')

                # Selected zone detail
                with ui.element('div').classes('geo-card'):
                    ui.html('<div class="geo-header">Selected Zone Detail</div>')
                    stat_cols = [
                        {'name':'metric','label':'Metric', 'field':'metric','align':'left'},
                        {'name':'value', 'label':'Value',  'field':'value', 'align':'right'},
                        {'name':'status','label':'Status', 'field':'status','align':'left'},
                    ]
                    charts['stats_table'] = ui.table(
                        columns=stat_cols, rows=[], row_key='metric'
                    ).classes('w-full')
                    charts['stats_table'].add_slot('body-cell-status', r'''
                        <q-td :props="props">
                          <span :style="{
                            color: props.row.status.startsWith('✓') ? '#10b981'
                                 : props.row.status.startsWith('⚠') ? '#f59e0b'
                                 : '#ffd700'}">
                            {{ props.row.status }}
                          </span>
                        </q-td>
                    ''')

    # ── Footer ────────────────────────────────────────────────────────────────
    with ui.footer().style('background:#111827;border-top:1px solid #2d3a4f;'
                           'height:28px;padding:0 16px'):
        with ui.row().classes('items-center gap-6 h-full'):
            with ui.row().classes('items-center gap-1'):
                ui.html('<div style="width:6px;height:6px;background:#10b981;'
                        'border-radius:50%"></div>')
                ui.label('Ready').classes('text-[10px] text-gray-500')
            ui.label('GeoConvention 2010 methodology  ·  Synthetic Volve-like dataset  ·  '
                     'NiceGUI + ECharts + Plotly').classes('text-[10px] text-gray-600')
            ui.label('LayerOptimum Pro v3.0').classes('text-[10px] text-gray-600')

    # ── Deferred initial population (gives browser time to init ECharts) ──────
    ui.timer(0.4, upd_xplot,     once=True)
    ui.timer(0.6, upd_layer,     once=True)
    ui.timer(1.2, upd_stats,     once=True)
    ui.timer(0.8, upd_vario,     once=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ui.run(
        host='127.0.0.1',
        port=8060,
        title='LayerOptimum Pro',
        dark=True,
        reload=False,
        show=True,
        favicon='⛽',
    )
