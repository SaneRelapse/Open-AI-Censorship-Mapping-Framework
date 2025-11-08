# restriction_dashboard.py
# Dash dashboard with incident reports + top-restrictive-theme hover info + OpenAI release timeline panel
# Run: python restriction_dashboard.py
# Requirements:
#   pip install dash plotly pandas dash-bootstrap-components

import io
from datetime import datetime
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# -------------------------
# SAMPLE DATA (prototype)
# -------------------------
SAMPLE_DATA = {
    "USA": {
        "name": "United States",
        "coords": [-98.5, 39.8],
        "timeline": [
            {
                "date": "2023-01-01",
                "scores": {"Optics": 10, "Occultism": 40, "Cybersecurity": 30, "Machine Vision": 20, "Law": 15, "Medicine": 5, "Firearms Research": 25},
                "incidents": [
                    {"theme": "Cybersecurity", "prompt": "User requested step-by-step exploit code (redacted)", "redacted": True, "reason": "Actionable exploit instructions", "date": "2023-01-01"},
                    {"theme": "Occultism", "prompt": "Discussion on ritual symbolism (allowed)", "redacted": False, "reason": "Non-actionable cultural discussion", "date": "2023-01-01"}
                ]
            },
            {
                "date": "2024-01-01",
                "scores": {"Optics": 12, "Occultism": 42, "Cybersecurity": 35, "Machine Vision": 25, "Law": 20, "Medicine": 10, "Firearms Research": 28},
                "incidents": [
                    {"theme": "Firearms Research", "prompt": "High-level history of firearm design (allowed)", "redacted": False, "reason": "General historical info", "date": "2024-01-05"}
                ]
            },
            {
                "date": "2025-01-01",
                "scores": {"Optics": 8, "Occultism": 35, "Cybersecurity": 40, "Machine Vision": 30, "Law": 18, "Medicine": 12, "Firearms Research": 30},
                "incidents": [
                    {"theme": "Cybersecurity", "prompt": "Request for exploit examples (redacted)", "redacted": True, "reason": "Actionable cybersecurity exploitation", "date": "2025-01-20"},
                    {"theme": "Medicine", "prompt": "High-level symptoms discussion (allowed)", "redacted": False, "reason": "Non-diagnostic general info", "date": "2025-01-21"}
                ]
            }
        ],
        "notes": "Prototype sample — replace with verified sources and sanitized incident text."
    },
    "CHN": {
        "name": "China",
        "coords": [104.2, 35.9],
        "timeline": [
            {
                "date": "2023-01-01",
                "scores": {"Optics": 60, "Occultism": 70, "Cybersecurity": 85, "Machine Vision": 80, "Law": 90, "Medicine": 75, "Firearms Research": 95},
                "incidents": [
                    {"theme": "Law", "prompt": "Request for commentary on banned political speech (redacted)", "redacted": True, "reason": "Political content restricted", "date": "2023-02-02"},
                    {"theme": "Firearms Research", "prompt": "How to build an improvised weapon (redacted)", "redacted": True, "reason": "Violent or weapon construction content", "date": "2023-03-03"}
                ]
            },
            {
                "date": "2024-01-01",
                "scores": {"Optics": 65, "Occultism": 75, "Cybersecurity": 88, "Machine Vision": 82, "Law": 92, "Medicine": 78, "Firearms Research": 96},
                "incidents": []
            },
            {
                "date": "2025-01-01",
                "scores": {"Optics": 62, "Occultism": 73, "Cybersecurity": 90, "Machine Vision": 85, "Law": 95, "Medicine": 80, "Firearms Research": 97},
                "incidents": []
            }
        ],
        "notes": "Sample high-censorship scores for demonstration."
    },
    "RUS": {
        "name": "Russia",
        "coords": [105.3, 61.5],
        "timeline": [
            {
                "date": "2023-01-01",
                "scores": {"Optics": 40, "Occultism": 60, "Cybersecurity": 70, "Machine Vision": 65, "Law": 80, "Medicine": 50, "Firearms Research": 60},
                "incidents": [
                    {"theme": "Cybersecurity", "prompt": "Request for targeted intrusion code (redacted)", "redacted": True, "reason": "Actionable intrusion instructions", "date": "2023-05-12"}
                ]
            },
            {
                "date": "2024-01-01",
                "scores": {"Optics": 42, "Occultism": 62, "Cybersecurity": 75, "Machine Vision": 68, "Law": 82, "Medicine": 52, "Firearms Research": 62},
                "incidents": []
            },
            {
                "date": "2025-01-01",
                "scores": {"Optics": 44, "Occultism": 58, "Cybersecurity": 78, "Machine Vision": 70, "Law": 85, "Medicine": 55, "Firearms Research": 65},
                "incidents": []
            }
        ],
        "notes": "Sample dataset entry."
    },
    "CAN": {
        "name": "Canada",
        "coords": [-106.3, 56.1],
        "timeline": [
            {
                "date": "2023-01-01",
                "scores": {"Optics": 8, "Occultism": 30, "Cybersecurity": 25, "Machine Vision": 20, "Law": 18, "Medicine": 8, "Firearms Research": 40},
                "incidents": [
                    {"theme": "Firearms Research", "prompt": "Discussion of licensing requirements (allowed)", "redacted": False, "reason": "Legal compliance info", "date": "2023-06-01"}
                ]
            },
            {
                "date": "2024-01-01",
                "scores": {"Optics": 9, "Occultism": 33, "Cybersecurity": 28, "Machine Vision": 22, "Law": 20, "Medicine": 9, "Firearms Research": 42},
                "incidents": []
            },
            {
                "date": "2025-01-01",
                "scores": {"Optics": 7, "Occultism": 28, "Cybersecurity": 30, "Machine Vision": 24, "Law": 19, "Medicine": 10, "Firearms Research": 45},
                "incidents": []
            }
        ],
        "notes": "Sample values."
    },
    "GBR": {
        "name": "United Kingdom",
        "coords": [-3.4, 55.4],
        "timeline": [
            {
                "date": "2023-01-01",
                "scores": {"Optics": 7, "Occultism": 35, "Cybersecurity": 30, "Machine Vision": 28, "Law": 22, "Medicine": 10, "Firearms Research": 50},
                "incidents": []
            },
            {
                "date": "2024-01-01",
                "scores": {"Optics": 6, "Occultism": 36, "Cybersecurity": 32, "Machine Vision": 30, "Law": 24, "Medicine": 12, "Firearms Research": 52},
                "incidents": []
            },
            {
                "date": "2025-01-01",
                "scores": {"Optics": 5, "Occultism": 34, "Cybersecurity": 33, "Machine Vision": 32, "Law": 25, "Medicine": 13, "Firearms Research": 55},
                "incidents": []
            }
        ],
        "notes": "Sample values."
    }
}

THEMES = ["Optics", "Occultism", "Cybersecurity", "Machine Vision", "Law", "Medicine", "Firearms Research"]

# -------------------------
# OpenAI release timeline (prototype)
# -------------------------
OPENAI_RELEASES = [
    {"date": "2025-11-08", "title": "OpenAI updates moderation: Medicine & Law restrictions", "summary": "OpenAI updated moderation policy increasing restrictions on medical and legal advice content in certain contexts.", "url": "https://example.com/openai-release-2025-11-08"},
    {"date": "2025-06-10", "title": "OpenAI tightens cybersecurity content handling", "summary": "Policy clarifications on handling cybersecurity-related content and exploit requests.", "url": "https://example.com/openai-release-2025-06-10"},
    {"date": "2024-12-01", "title": "OpenAI updates nation-level policy handling", "summary": "Platform clarified handling of jurisdictional legal requests and region-specific limitations.", "url": "https://example.com/openai-release-2024-12-01"}
]
for r in OPENAI_RELEASES:
    r["date_ts"] = pd.to_datetime(r["date"])
OPENAI_RELEASES = sorted(OPENAI_RELEASES, key=lambda x: x["date_ts"], reverse=True)

# -------------------------
# Build tidy DataFrame + incident table from SAMPLE_DATA
# -------------------------
def build_dataframe_and_incidents(data):
    rows = []
    incidents_rows = []
    for iso3, r in data.items():
        for t in r["timeline"]:
            date = pd.to_datetime(t["date"])
            scores = t.get("scores", {})
            for theme, score in scores.items():
                rows.append({
                    "iso3": iso3,
                    "country": r["name"],
                    "lon": r["coords"][0],
                    "lat": r["coords"][1],
                    "date": date,
                    "theme": theme,
                    "score": score,
                    "notes": r.get("notes", "")
                })
            for inc in t.get("incidents", []):
                incidents_rows.append({
                    "iso3": iso3,
                    "country": r["name"],
                    "date": date,
                    "theme": inc.get("theme"),
                    "prompt": inc.get("prompt"),
                    "redacted": bool(inc.get("redacted", False)),
                    "reason": inc.get("reason", ""),
                    "inc_date": pd.to_datetime(inc.get("date")) if inc.get("date") else date
                })
    df = pd.DataFrame(rows)
    incidents_df = pd.DataFrame(incidents_rows)
    return df, incidents_df

DF, INCIDENTS_DF = build_dataframe_and_incidents(SAMPLE_DATA)
DATES = list(pd.to_datetime(DF["date"].unique()).sort_values())

# -------------------------
# Dash app
# -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server

app.layout = dbc.Container(fluid=True, children=[
    dbc.Row([
        dbc.Col(html.H2("ChatGPT Restriction Atlas — Dash (Prototype)"), width=9),
        dbc.Col(dbc.Button("Export CSV", id="export-btn", color="secondary", size="sm"), width=3, style={"textAlign": "right"})
    ], align="center", className="my-2"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dcc.Input(id="search-box", placeholder="Search country (name) ...", debounce=True, type="text", style={"width":"100%"}), md=4),
                        dbc.Col(dcc.Dropdown(id="theme-dropdown", options=[{"label": t, "value": t} for t in THEMES], value="Cybersecurity", clearable=False), md=4),
                        dbc.Col(dcc.Slider(id="date-slider", min=0, max=len(DATES)-1, value=len(DATES)-1,
                                           marks={i: d.strftime("%Y-%m-%d") for i, d in enumerate(DATES)}, tooltip={"placement":"bottom"}, updatemode="mouseup"), md=4),
                    ], className="mb-2"),
                    dcc.Graph(id="choropleth-map", config={"displayModeBar": True, "scrollZoom": True}, style={"height": "640px"})
                ])
            ], style={"height": "100%"})
        ], md=7),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Selected region"),
                    html.Div(id="selected-region-card", children=[html.P("Click a country on the map to see details, or search and hit Enter.")]),
                    html.Hr(),
                    html.H6("Per-topic scores (selected date)"),
                    html.Div(id="per-topic-scores", style={"marginBottom":"8px"}),
                    html.Hr(),
                    html.H6("Incident Reports (selected date/country)"),
                    html.Div(id="incident-reports", children=[html.P("Click a country to view incident reports. Reports are shown sanitized and anonymized.")]),
                    html.Hr(),
                    html.Details([html.Summary("Data sources & notes"),
                                  html.P("Prototype sample data. Replace SAMPLE_DATA with real, verified transparency reports or logs. ALWAYS sanitize prompts to remove operationally harmful details."),
                                  html.Ul([html.Li("Include: source, date, confidence score, evidence link"),
                                           html.Li("Sanitize blocked prompt text; remove step-by-step instructions")])]),
                    html.Hr(),
                    dbc.Button("Toggle theme (light/dark)", id="toggle-theme", color="light", size="sm"),
                    html.Div(id="hidden-download", style={"display": "none"})
                ])
            ])
        ], md=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("OpenAI Release Timeline"),
                    dcc.Dropdown(
                        id="release-dropdown",
                        options=[{"label": f"{r['date']} — {r['title']}", "value": idx} for idx, r in enumerate(OPENAI_RELEASES)],
                        placeholder="Select a release...",
                        clearable=True,
                        style={"fontSize":"12px"}
                    ),
                    html.Div(id="release-details", style={"marginTop":"8px", "fontSize":"13px"}),
                    html.Hr(),
                    html.P("Recent releases (click to open):", style={"fontSize":"12px", "opacity":0.85}),
                    html.Ul([
                        html.Li(html.A(f"{r['date']} — {r['title']}", href=r['url'], target="_blank", rel="noopener noreferrer", style={"color":"#9ae6b4" if idx==0 else None}))
                        for idx, r in enumerate(OPENAI_RELEASES)
                    ], style={"paddingLeft":"16px", "maxHeight":"420px", "overflowY":"auto"})
                ])
            ])
        ], md=2)
    ], className="g-2"),

    dcc.Store(id="clicked-iso", storage_type="memory"),
    dcc.Download(id="download-data")
], style={"padding": "12px"})

# -------------------------
# Helpers
# -------------------------
def prepare_choropleth_df(df, incidents_df, theme, date_idx, search_query=""):
    date = DATES[date_idx]
    df_date = df[df["date"] == date]
    if search_query:
        df_date = df_date[df_date["country"].str.contains(search_query, case=False, na=False)]
    df_sel = df_date[df_date["theme"] == theme].copy()
    pivot = df_date.pivot_table(index=["iso3", "country", "lon", "lat"], columns="theme", values="score").fillna(-1)
    def top_themes_from_row(row):
        s = row.sort_values(ascending=False)
        top = [str(idx) for idx, val in s.items() if val >= 0][:2]
        return ", ".join(top) if top else "—"
    pivot["top_themes"] = pivot.apply(top_themes_from_row, axis=1)
    inc_date = incidents_df[incidents_df["date"] == date]
    inc_counts = inc_date.groupby("iso3").size().to_dict()
    if not df_sel.empty:
        df_sel = df_sel.merge(pivot[["top_themes"]].reset_index(), on=["iso3", "country", "lon", "lat"], how="left")
    else:
        df_sel = pivot.reset_index()[["iso3", "country", "lon", "lat", "top_themes"]]
        df_sel["theme"] = theme
        df_sel["score"] = np.nan
    df_sel["incident_count"] = df_sel["iso3"].map(inc_counts).fillna(0).astype(int)
    df_sel["hover_summary"] = df_sel.apply(lambda r: f"{r['country']}<br>{theme}: {r['score'] if pd.notna(r['score']) else 'N/A'}<br>Top topics: {r.get('top_themes','—')}<br>Incidents: {r['incident_count']}", axis=1)
    df_sel["score_clamped"] = df_sel["score"]
    return df_sel

def build_choropleth_figure(df_slice, theme, date_idx, light_theme=False, min_score=0, max_score=100):
    if df_slice.empty:
        fig = px.choropleth(locations=[], locationmode="ISO-3", scope="world")
        fig.update_layout(height=640, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    color_continuous_scale = [(0.0, "#44ea6b"), (0.5, "#ffd24b"), (1.0, "#ff3b3b")]
    fig = px.choropleth(
        df_slice,
        locations="iso3",
        color="score_clamped",
        hover_name="country",
        hover_data={"score_clamped": False, "iso3": True, "incident_count": True, "top_themes": True},
        color_continuous_scale=color_continuous_scale,
        range_color=(min_score, max_score),
        labels={"score_clamped": f"{theme} restriction score"},
        locationmode="ISO-3",
        projection="natural earth"
    )
    fig.update_traces(
        hovertemplate="%{customdata[0]}<br>Top topics: %{customdata[2]}<br>Incidents: %{customdata[1]}<extra></extra>",
        customdata=df_slice[["iso3", "incident_count", "top_themes"]].values
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="0 (open) → 100 (blocked)"),
        template="plotly_dark" if not light_theme else "plotly_white",
        title=f"{theme} — {DATES[date_idx].strftime('%Y-%m-%d')}"
    )
    return fig

def compute_per_topic_scores_for_iso(df, iso3, date_idx):
    date = DATES[date_idx]
    df_date = df[df["date"] == date]
    if iso3 is None:
        avgs = df_date.groupby("theme")["score"].mean().round(1).to_dict()
    else:
        df_iso = df_date[df_date["iso3"] == iso3]
        avgs = df_iso.set_index("theme")["score"].to_dict()
    out = {t: round(float(avgs.get(t, np.nan)), 1) if not pd.isna(avgs.get(t, np.nan)) else "N/A" for t in THEMES}
    return out

def get_incidents_for_iso_and_date(incidents_df, iso3, date_idx):
    date = DATES[date_idx]
    df = incidents_df[(incidents_df["iso3"] == iso3) & (incidents_df["date"] == date)]
    if df.empty:
        return []
    df_sorted = df.sort_values("inc_date", ascending=False)
    out = df_sorted.to_dict("records")
    for i in out:
        i["inc_date"] = pd.to_datetime(i["inc_date"]).strftime("%Y-%m-%d")
    return out

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output("choropleth-map", "figure"),
    Input("theme-dropdown", "value"),
    Input("date-slider", "value"),
    Input("search-box", "value"),
    Input("toggle-theme", "n_clicks"),
    State("choropleth-map", "figure"),
)
def update_map(theme, date_idx, search_query, toggle_clicks, prior_fig):
    light_theme = (toggle_clicks or 0) % 2 == 1
    df_slice = prepare_choropleth_df(DF, INCIDENTS_DF, theme, date_idx, search_query=(search_query or ""))
    fig = build_choropleth_figure(df_slice, theme, date_idx, light_theme=light_theme)
    return fig

@app.callback(
    Output("clicked-iso", "data"),
    Input("choropleth-map", "clickData"),
    prevent_initial_call=True
)
def map_click(clickData):
    if not clickData:
        return dash.no_update
    point = clickData.get("points", [{}])[0]
    iso = point.get("location")
    return iso

@app.callback(
    Output("selected-region-card", "children"),
    Output("per-topic-scores", "children"),
    Output("incident-reports", "children"),
    Input("clicked-iso", "data"),
    Input("date-slider", "value"),
    Input("theme-dropdown", "value")
)
def update_sidebar(clicked_iso, date_idx, theme):
    iso = clicked_iso
    if not iso:
        scores = compute_per_topic_scores_for_iso(DF, None, date_idx)
        card = html.Div([html.Strong("Global sample (no country selected)"), html.P(f"Date: {DATES[date_idx].strftime('%Y-%m-%d')}"), html.P("Hint: click a country on the map or search above.")])
        incidents_panel = html.P("Click a country to view incident reports.")
    else:
        country_rows = DF[(DF["iso3"] == iso) & (DF["date"] == DATES[date_idx])]
        if country_rows.empty:
            card = html.Div([html.Strong("No data for this country at selected date."), html.P(f"ISO3: {iso}")])
            incidents_panel = html.P("No incident records for this country/date.")
            scores = compute_per_topic_scores_for_iso(DF, None, date_idx)
        else:
            country_name = country_rows.iloc[0]["country"]
            notes = country_rows.iloc[0]["notes"]
            card = html.Div([html.H4(country_name), html.P(f"ISO3: {iso}"), html.P(f"Date: {DATES[date_idx].strftime('%Y-%m-%d')}"), html.P("Quick sample blocked prompts (sanitized/anonymized):")])
            scores = compute_per_topic_scores_for_iso(DF, iso, date_idx)
            incident_list = get_incidents_for_iso_and_date(INCIDENTS_DF, iso, date_idx)
            if not incident_list:
                incidents_panel = html.P("No incident reports for this country/date.")
            else:
                items = []
                for idx, inc in enumerate(incident_list):
                    status = "Redacted" if inc.get("redacted") else "Shown"
                    header = f"{inc.get('theme')} — {inc.get('inc_date')} — {status}"
                    body = [html.P(html.Span("Prompt (sanitized): "), style={"fontWeight": "600"}), html.P(inc.get("prompt")), html.P(html.Span("Reason: "), style={"fontWeight": "600"}), html.P(inc.get("reason"))]
                    items.append(dbc.Card([dbc.CardHeader(html.Span(header)), dbc.Collapse(dbc.CardBody(body), id={"type": "inc-collapse", "index": f"{iso}-{idx}"}, is_open=False)], className="mb-2"))
                incidents_panel = html.Div([html.P(f"{len(items)} incident(s) — click to expand details (sanitized)"), html.Div(items)])
    per_topic_children = []
    for t, v in scores.items():
        color = "success"
        try:
            vn = float(v)
            if vn >= 70:
                color = "danger"
            elif vn >= 40:
                color = "warning"
        except Exception:
            color = "secondary"
        per_topic_children.append(html.Div([html.Span(t), dbc.Badge(str(v), color=color, className="float-end")], style={"padding":"6px 0", "borderBottom":"1px solid rgba(255,255,255,0.04)"}))
    return card, per_topic_children, incidents_panel

@app.callback(
    Output({"type": "inc-collapse", "index": dash.ALL}, "is_open"),
    Input({"type": "inc-collapse", "index": dash.ALL}, "n_clicks"),
    State({"type": "inc-collapse", "index": dash.ALL}, "is_open"),
    prevent_initial_call=True
)
def toggle_incidents(n_clicks_list, is_open_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    new_states = list(is_open_list)
    for i, nk in enumerate(n_clicks_list):
        if nk and nk > 0:
            new_states[i] = not new_states[i]
    return new_states

@app.callback(
    Output("download-data", "data"),
    Input("export-btn", "n_clicks"),
    prevent_initial_call=True
)
def export_csv(n):
    buf = io.StringIO()
    buf.write("# restrictions_snapshot\n")
    DF.to_csv(buf, index=False)
    buf.write("\n# incidents_snapshot\n")
    INCIDENTS_DF.to_csv(buf, index=False)
    buf.seek(0)
    return dcc.send_string(buf.getvalue(), filename=f"censorship_snapshot_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv")

@app.callback(
    Output("release-details", "children"),
    Input("release-dropdown", "value")
)
def show_release_details(selected_idx):
    if selected_idx is None:
        r = OPENAI_RELEASES[0] if OPENAI_RELEASES else None
    else:
        r = OPENAI_RELEASES[int(selected_idx)]
    if not r:
        return html.P("No release selected.")
    return html.Div([html.Div(html.Strong(f"{r['date']} — {r['title']}")), html.Div(r["summary"], style={"fontSize":"12px", "marginTop":"6px"}), html.Div(html.A("Open publication", href=r["url"], target="_blank", rel="noopener noreferrer"), style={"marginTop":"8px"})])

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    print("Starting Dash app on http://127.0.0.1:8050/")
    app.run(debug=True)



