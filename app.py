# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import scipy.stats as stats

# --- APP CONFIG ---
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.CYBORG],
                suppress_callback_exceptions=True)
server = app.server

# Paleta industrial: azul acero + naranja alerta + fondo oscuro
colors = {
    'background': '#0D1117',
    'card_bg': '#161B22',
    'border': '#21262D',
    'accent': '#1F6FEB',       # azul eléctrico
    'warning': '#F85149',      # rojo falla
    'success': '#3FB950',      # verde normal
    'recovering': '#D29922',   # amarillo recuperación
    'text': '#E6EDF3',
    'subtext': '#8B949E',
    'grid': '#21262D'
}

STATUS_COLORS = {
    'NORMAL': colors['success'],
    'BROKEN': colors['warning'],
    'RECOVERING': colors['recovering']
}

# --- DATOS SIMULADOS (estructura real del dataset sensor pump) ---
np.random.seed(42)
n = 220320  # ~5 meses de datos minuto a minuto

timestamps = pd.date_range(start='2018-04-01', periods=n, freq='T')

# Simular 52 sensores con patrones realistas
sensor_data = {}
for i in range(52):
    base = np.random.uniform(10, 100)
    noise = np.random.normal(0, base * 0.05, n)
    trend = np.linspace(0, np.random.uniform(-5, 5), n)
    sensor_data[f'sensor_{i:02d}'] = base + noise + trend

df = pd.DataFrame(sensor_data, index=timestamps)
df.index.name = 'timestamp'

# Machine status
status = np.array(['NORMAL'] * n, dtype=object)
broken_idx = np.random.choice(range(10000, n-5000), size=12, replace=False)
for idx in broken_idx:
    status[idx:idx+60] = 'BROKEN'
    status[idx+60:idx+300] = 'RECOVERING'

df['machine_status'] = status
df = df.reset_index()

# Algunos NaN realistas
for col in ['sensor_00', 'sensor_15', 'sensor_50']:
    df.loc[np.random.choice(df.index, size=200, replace=False), col] = np.nan

# Stats globales
sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
status_counts = df['machine_status'].value_counts()
total_broken = int(status_counts.get('BROKEN', 0))
total_normal = int(status_counts.get('NORMAL', 0))
total_recovering = int(status_counts.get('RECOVERING', 0))
missing_pct = df[sensor_cols].isna().mean().mean() * 100
sensors_with_missing = int((df[sensor_cols].isna().sum() > 0).sum())

# Muestra para gráficos temporales (1 cada 60 pts)
df_sample = df.iloc[::60].copy()

# --- LAYOUT ---
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'fontFamily': 'Segoe UI, sans-serif'}, children=[

    dbc.Container(fluid=True, style={'padding': '0 32px'}, children=[

        # HEADER
        html.Div(style={'padding': '36px 0 20px', 'borderBottom': f'1px solid {colors["border"]}'}, children=[
            dbc.Row(align='center', children=[
                dbc.Col(width='auto', children=[
                    html.Div('⚙️', style={'fontSize': '48px'})
                ]),
                dbc.Col(children=[
                    html.H1("EDA — Bomba de Agua Industrial",
                            style={'color': colors['text'], 'fontWeight': '700', 'marginBottom': '4px', 'fontSize': '28px'}),
                    html.P("Pump Sensor Data · Mantenimiento Predictivo · Abril–Agosto 2018",
                           style={'color': colors['subtext'], 'margin': 0, 'fontSize': '14px'}),
                ])
            ])
        ]),

        # TABS
        dbc.Tabs(id='main-tabs', active_tab='tab-overview', style={'marginTop': '24px'}, children=[
            dbc.Tab(label='📊 Overview', tab_id='tab-overview'),
            dbc.Tab(label='📈 Series de Tiempo', tab_id='tab-time'),
            dbc.Tab(label='🔍 Distribuciones', tab_id='tab-dist'),
            dbc.Tab(label='🔗 Correlaciones', tab_id='tab-corr'),
            dbc.Tab(label='⚠️ Anomalías', tab_id='tab-anomaly'),
            dbc.Tab(label='📋 Inferencia', tab_id='tab-stats'),
        ]),

        html.Div(id='tab-content', style={'paddingTop': '28px', 'paddingBottom': '48px'}),
    ])
])

# Helper card
def metric_card(title, value, subtitle='', color=colors['accent'], icon=''):
    return dbc.Col(md=3, children=[
        html.Div(style={
            'backgroundColor': colors['card_bg'], 'borderRadius': '10px',
            'border': f'1px solid {colors["border"]}', 'padding': '20px',
            'borderTop': f'3px solid {color}'
        }, children=[
            html.Div(f"{icon} {title}", style={'color': colors['subtext'], 'fontSize': '12px', 'textTransform': 'uppercase', 'letterSpacing': '1px'}),
            html.Div(str(value), style={'color': color, 'fontSize': '32px', 'fontWeight': '700', 'margin': '6px 0'}),
            html.Div(subtitle, style={'color': colors['subtext'], 'fontSize': '12px'}),
        ])
    ])

def section_title(text):
    return html.H5(text, style={'color': colors['text'], 'marginBottom': '16px', 'marginTop': '32px',
                                 'borderLeft': f'3px solid {colors["accent"]}', 'paddingLeft': '12px'})

# --- CALLBACKS ---
@app.callback(Output('tab-content', 'children'), Input('main-tabs', 'active_tab'))
def render_tab(tab):

    # ── TAB 1: OVERVIEW ──────────────────────────────────────────────
    if tab == 'tab-overview':
        # Pie chart
        pie = go.Figure(go.Pie(
            labels=list(status_counts.index),
            values=list(status_counts.values),
            marker_colors=[STATUS_COLORS.get(s, '#888') for s in status_counts.index],
            hole=0.55,
            textinfo='label+percent',
            textfont_color=colors['text']
        ))
        pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'], margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(font_color=colors['text'], bgcolor='rgba(0,0,0,0)')
        )

        # Missing por sensor (top 15)
        missing_by_sensor = df[sensor_cols].isna().sum().sort_values(ascending=False).head(15)
        bar_missing = go.Figure(go.Bar(
            x=missing_by_sensor.index,
            y=missing_by_sensor.values,
            marker_color=colors['warning'],
            opacity=0.85
        ))
        bar_missing.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'], margin=dict(t=20, b=40, l=40, r=10),
            xaxis=dict(gridcolor=colors['grid'], color=colors['subtext']),
            yaxis=dict(gridcolor=colors['grid'], color=colors['subtext'], title='Valores faltantes')
        )

        return html.Div([
            section_title("Métricas Generales"),
            dbc.Row([
                metric_card("Registros Totales", f"{len(df):,}", "5 meses · resolución 1 min", colors['accent'], '📁'),
                metric_card("Sensores", "52", "sensor_00 a sensor_51", colors['accent'], '🔌'),
                metric_card("Fallas (BROKEN)", f"{total_broken:,}", f"{total_broken/len(df)*100:.2f}% del tiempo", colors['warning'], '💥'),
                metric_card("En Recuperación", f"{total_recovering:,}", f"{total_recovering/len(df)*100:.2f}% del tiempo", colors['recovering'], '🔧'),
            ], className='g-3'),

            dbc.Row([
                metric_card("% Datos Faltantes", f"{missing_pct:.2f}%", "promedio entre sensores", colors['recovering'], '❓'),
                metric_card("Sensores con NaN", str(sensors_with_missing), "de 52 sensores", colors['recovering'], '⚠️'),
                metric_card("Período", "Abr–Ago 2018", "220,320 lecturas", colors['success'], '📅'),
                metric_card("Variable Objetivo", "machine_status", "NORMAL / BROKEN / RECOVERING", colors['success'], '🎯'),
            ], className='g-3 mt-1'),

            section_title("Distribución de Estados Operativos"),
            dbc.Row([
                dbc.Col(md=5, children=[dcc.Graph(figure=pie, config={'displayModeBar': False})]),
                dbc.Col(md=7, children=[
                    html.Div(style={'backgroundColor': colors['card_bg'], 'borderRadius': '10px',
                                    'border': f'1px solid {colors["border"]}', 'padding': '24px', 'height': '100%'}, children=[
                        html.H6("Descripción del Dataset", style={'color': colors['text'], 'fontWeight': '600', 'marginBottom': '16px'}),
                        html.P("Este dataset contiene series de tiempo de 52 sensores instalados en una bomba de agua industrial monitoreada durante 5 meses (Abril–Agosto 2018). Cada fila representa una lectura simultánea de todos los sensores en un minuto determinado.",
                               style={'color': colors['subtext'], 'lineHeight': '1.7', 'fontSize': '14px'}),
                        html.Hr(style={'borderColor': colors['border']}),
                        html.Div([
                            html.Span("● NORMAL", style={'color': colors['success'], 'fontWeight': '600', 'marginRight': '16px'}),
                            html.Span("● BROKEN", style={'color': colors['warning'], 'fontWeight': '600', 'marginRight': '16px'}),
                            html.Span("● RECOVERING", style={'color': colors['recovering'], 'fontWeight': '600'}),
                        ]),
                        html.P("El objetivo es identificar patrones en los sensores para anticipar fallas (BROKEN) o detectar periodos de recuperación (RECOVERING), construyendo un sistema de mantenimiento predictivo.",
                               style={'color': colors['subtext'], 'lineHeight': '1.7', 'fontSize': '14px', 'marginTop': '12px'}),
                    ])
                ])
            ]),

            section_title("Valores Faltantes por Sensor (Top 15)"),
            dcc.Graph(figure=bar_missing, config={'displayModeBar': False}),
        ])

    # ── TAB 2: SERIES DE TIEMPO ───────────────────────────────────────
    elif tab == 'tab-time':
        sensors_to_show = ['sensor_00', 'sensor_04', 'sensor_10', 'sensor_15', 'sensor_23']

        # Status timeline
        status_numeric = df_sample['machine_status'].map({'NORMAL': 0, 'BROKEN': 2, 'RECOVERING': 1})
        fig_status = go.Figure()
        for s, val, col in [('NORMAL', 0, colors['success']), ('BROKEN', 2, colors['warning']), ('RECOVERING', 1, colors['recovering'])]:
            mask = df_sample['machine_status'] == s
            fig_status.add_trace(go.Scatter(
                x=df_sample.loc[mask, 'timestamp'], y=status_numeric[mask],
                mode='markers', marker=dict(color=col, size=3, opacity=0.7),
                name=s
            ))
        fig_status.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'], height=160, margin=dict(t=10, b=30, l=60, r=10),
            xaxis=dict(gridcolor=colors['grid'], color=colors['subtext']),
            yaxis=dict(gridcolor=colors['grid'], tickvals=[0,1,2], ticktext=['NORMAL','RECOV.','BROKEN'], color=colors['subtext']),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=1.2),
            showlegend=True
        )

        # Sensores seleccionados
        fig_sensors = make_subplots(rows=len(sensors_to_show), cols=1, shared_xaxes=True,
                                     vertical_spacing=0.03)
        palette = [colors['accent'], '#A371F7', '#FF9800', '#00BCD4', '#E91E63']
        for i, (sensor, col) in enumerate(zip(sensors_to_show, palette)):
            fig_sensors.add_trace(go.Scatter(
                x=df_sample['timestamp'], y=df_sample[sensor],
                mode='lines', line=dict(color=col, width=1),
                name=sensor
            ), row=i+1, col=1)
            fig_sensors.update_yaxes(title_text=sensor, row=i+1, col=1,
                                      gridcolor=colors['grid'], color=colors['subtext'], title_font_size=10)
        fig_sensors.update_xaxes(gridcolor=colors['grid'], color=colors['subtext'])
        fig_sensors.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'], height=500, margin=dict(t=10, b=40, l=80, r=10),
            showlegend=False
        )

        return html.Div([
            section_title("Estado Operativo a lo largo del tiempo"),
            dcc.Graph(figure=fig_status, config={'displayModeBar': False}),
            section_title("Series de Tiempo — Sensores Seleccionados"),
            dcc.Graph(figure=fig_sensors, config={'displayModeBar': False}),
        ])

    # ── TAB 3: DISTRIBUCIONES ─────────────────────────────────────────
    elif tab == 'tab-dist':
        # Selector de sensor
        sensor_options = [{'label': s, 'value': s} for s in sensor_cols[:20]]

        # Stats desc
        desc = df[sensor_cols].describe().T[['mean','std','min','50%','max']]
        desc.columns = ['Media', 'Std', 'Mín', 'Mediana', 'Máx']
        desc = desc.round(3).reset_index().rename(columns={'index': 'Sensor'})

        table_header = [html.Thead(html.Tr([html.Th(c, style={'color': colors['accent'], 'borderColor': colors['border']}) for c in desc.columns]))]
        table_rows = [html.Tr([html.Td(row[c], style={'color': colors['text'], 'borderColor': colors['border'], 'fontSize': '13px'}) for c in desc.columns]) for _, row in desc.head(20).iterrows()]
        table_body = [html.Tbody(table_rows)]

        return html.Div([
            section_title("Estadística Descriptiva (primeros 20 sensores)"),
            html.Div(style={'backgroundColor': colors['card_bg'], 'borderRadius': '10px',
                            'border': f'1px solid {colors["border"]}', 'padding': '16px', 'overflowX': 'auto'}, children=[
                dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True,
                          style={'marginBottom': 0})
            ]),

            section_title("Distribución por Sensor"),
            html.Div(style={'marginBottom': '16px'}, children=[
                html.Label("Selecciona un sensor:", style={'color': colors['subtext'], 'fontSize': '13px', 'marginBottom': '8px', 'display': 'block'}),
                dcc.Dropdown(id='sensor-dropdown', options=sensor_options, value='sensor_00',
                             style={'width': '280px', 'backgroundColor': colors['card_bg'], 'color': '#000'},
                             clearable=False)
            ]),
            dcc.Graph(id='dist-graph', config={'displayModeBar': False}),
        ])

    # ── TAB 4: CORRELACIONES ──────────────────────────────────────────
    elif tab == 'tab-corr':
        # Correlación entre primeros 20 sensores
        corr_df = df[sensor_cols[:20]].dropna().corr()
        heatmap = go.Figure(go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns, y=corr_df.index,
            colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title='r', tickfont=dict(color=colors['text'])),
            text=corr_df.round(2).values,
            texttemplate='%{text}', textfont=dict(size=9)
        ))
        heatmap.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'], height=600,
            margin=dict(t=20, b=80, l=80, r=20),
            xaxis=dict(color=colors['subtext'], tickangle=-45),
            yaxis=dict(color=colors['subtext'])
        )

        # Top correlaciones
        corr_pairs = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool)).stack()
        top_corr = corr_pairs.abs().sort_values(ascending=False).head(10)
        top_df = pd.DataFrame({'Par de sensores': [f"{a} ↔ {b}" for a, b in top_corr.index],
                                'Correlación': corr_pairs.loc[top_corr.index].round(3).values})

        tbl_h = [html.Thead(html.Tr([html.Th(c, style={'color': colors['accent'], 'borderColor': colors['border']}) for c in top_df.columns]))]
        tbl_b = [html.Tbody([html.Tr([
            html.Td(row['Par de sensores'], style={'color': colors['text'], 'borderColor': colors['border']}),
            html.Td(str(row['Correlación']), style={'color': colors['success'] if abs(float(row['Correlación'])) > 0.7 else colors['text'], 'borderColor': colors['border']})
        ]) for _, row in top_df.iterrows()])]

        return html.Div([
            section_title("Mapa de Correlación — 20 Sensores"),
            dcc.Graph(figure=heatmap, config={'displayModeBar': False}),
            section_title("Top 10 Pares con Mayor Correlación"),
            html.Div(style={'backgroundColor': colors['card_bg'], 'borderRadius': '10px',
                            'border': f'1px solid {colors["border"]}', 'padding': '16px', 'maxWidth': '500px'}, children=[
                dbc.Table(tbl_h + tbl_b, bordered=True, hover=True, style={'marginBottom': 0})
            ])
        ])

    # ── TAB 5: ANOMALÍAS ─────────────────────────────────────────────
    elif tab == 'tab-anomaly':
        # Boxplot por estado
        sensor_for_box = 'sensor_04'
        fig_box = go.Figure()
        for status_val, col in STATUS_COLORS.items():
            subset = df[df['machine_status'] == status_val][sensor_for_box].dropna()
            fig_box.add_trace(go.Box(
                y=subset.sample(min(5000, len(subset)), random_state=42) if len(subset) > 5000 else subset,
                name=status_val, marker_color=col, line_color=col
            ))
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'], height=400, margin=dict(t=10, b=40, l=60, r=10),
            xaxis=dict(color=colors['subtext']),
            yaxis=dict(gridcolor=colors['grid'], color=colors['subtext'], title=sensor_for_box),
        )

        # IQR outliers por sensor
        outlier_counts = {}
        for s in sensor_cols[:20]:
            col_data = df[s].dropna()
            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            outliers = ((col_data < q1 - 1.5*iqr) | (col_data > q3 + 1.5*iqr)).sum()
            outlier_counts[s] = outliers

        oc_series = pd.Series(outlier_counts).sort_values(ascending=False)
        fig_outliers = go.Figure(go.Bar(
            x=oc_series.index, y=oc_series.values,
            marker_color=colors['warning'], opacity=0.85
        ))
        fig_outliers.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'], height=350, margin=dict(t=10, b=60, l=60, r=10),
            xaxis=dict(gridcolor=colors['grid'], color=colors['subtext'], tickangle=-30),
            yaxis=dict(gridcolor=colors['grid'], color=colors['subtext'], title='N° Outliers (IQR)'),
        )

        return html.Div([
            section_title(f"Distribución por Estado — {sensor_for_box}"),
            dcc.Graph(figure=fig_box, config={'displayModeBar': False}),
            section_title("Outliers por Sensor (Método IQR · primeros 20 sensores)"),
            dcc.Graph(figure=fig_outliers, config={'displayModeBar': False}),
        ])

    # ── TAB 6: INFERENCIA ────────────────────────────────────────────
    elif tab == 'tab-stats':
        # Kruskal-Wallis para cada sensor entre estados
        kw_results = []
        for s in sensor_cols[:15]:
            groups = [df[df['machine_status'] == st][s].dropna().values for st in ['NORMAL', 'BROKEN', 'RECOVERING']]
            groups = [g for g in groups if len(g) > 5]
            if len(groups) >= 2:
                stat, p = stats.kruskal(*groups)
                kw_results.append({'Sensor': s, 'Estadístico H': round(stat, 2), 'p-valor': round(p, 6),
                                    'Significativo (α=0.05)': '✅ Sí' if p < 0.05 else '❌ No'})

        kw_df = pd.DataFrame(kw_results)
        kw_header = [html.Thead(html.Tr([html.Th(c, style={'color': colors['accent'], 'borderColor': colors['border']}) for c in kw_df.columns]))]
        kw_rows = [html.Tr([
            html.Td(row[c], style={'color': colors['success'] if '✅' in str(row[c]) else colors['text'],
                                    'borderColor': colors['border'], 'fontSize': '13px'}) for c in kw_df.columns
        ]) for _, row in kw_df.iterrows()]
        kw_body = [html.Tbody(kw_rows)]

        # Violin plots comparativos
        fig_violin = go.Figure()
        for s, col in STATUS_COLORS.items():
            subset = df[df['machine_status'] == s]['sensor_04'].dropna()
            if len(subset) > 100:
                sample = subset.sample(min(3000, len(subset)), random_state=42)
                fig_violin.add_trace(go.Violin(y=sample, name=s, line_color=col,
                                                fillcolor=col, opacity=0.5, box_visible=True, meanline_visible=True))
        fig_violin.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color=colors['text'], height=380, margin=dict(t=10, b=40, l=60, r=10),
            xaxis=dict(color=colors['subtext']),
            yaxis=dict(gridcolor=colors['grid'], color=colors['subtext'], title='sensor_04'),
        )

        return html.Div([
            section_title("Prueba de Kruskal-Wallis — Diferencia entre Estados (primeros 15 sensores)"),
            html.P("H₀: La distribución del sensor es igual en los tres estados operativos. Un p-valor < 0.05 indica diferencias estadísticamente significativas.",
                   style={'color': colors['subtext'], 'fontSize': '13px', 'marginBottom': '16px'}),
            html.Div(style={'backgroundColor': colors['card_bg'], 'borderRadius': '10px',
                            'border': f'1px solid {colors["border"]}', 'padding': '16px', 'overflowX': 'auto'}, children=[
                dbc.Table(kw_header + kw_body, bordered=True, hover=True, responsive=True, style={'marginBottom': 0})
            ]),
            section_title("Violin Plot — sensor_04 por Estado"),
            dcc.Graph(figure=fig_violin, config={'displayModeBar': False}),
        ])

    return html.Div("Tab no encontrada", style={'color': colors['text']})


# Callback para distribución individual
@app.callback(Output('dist-graph', 'figure'), Input('sensor-dropdown', 'value'))
def update_dist(sensor):
    if not sensor:
        return go.Figure()
    data = df[sensor].dropna()
    sample = data.sample(min(10000, len(data)), random_state=42)

    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=['Histograma + KDE', 'Q-Q Plot'],
                         horizontal_spacing=0.1)

    # Histograma
    fig.add_trace(go.Histogram(x=sample, nbinsx=60, marker_color=colors['accent'],
                                opacity=0.7, name='Histograma'), row=1, col=1)

    # QQ
    qq = stats.probplot(sample)
    theoretical, ordered = qq[0]
    fig.add_trace(go.Scatter(x=theoretical, y=ordered, mode='markers',
                              marker=dict(color=colors['accent'], size=3, opacity=0.5),
                              name='Q-Q'), row=1, col=2)
    # Línea de referencia
    fig.add_trace(go.Scatter(x=[theoretical.min(), theoretical.max()],
                              y=[theoretical.min()*qq[1][0]+qq[1][1], theoretical.max()*qq[1][0]+qq[1][1]],
                              mode='lines', line=dict(color=colors['warning'], width=2), name='Normal ref.'), row=1, col=2)

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color=colors['text'], height=380, margin=dict(t=40, b=40, l=60, r=20),
        showlegend=False,
        annotations=[dict(font=dict(color=colors['subtext'], size=12)) for _ in range(2)]
    )
    fig.update_xaxes(gridcolor=colors['grid'], color=colors['subtext'])
    fig.update_yaxes(gridcolor=colors['grid'], color=colors['subtext'])
    return fig


if __name__ == '__main__':
    app.run(debug=True)
