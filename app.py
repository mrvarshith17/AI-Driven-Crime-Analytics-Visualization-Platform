"""
KSP Crime Intelligence & Visualization Platform
Streamlit app — reads pre-computed artifacts exported from the Kaggle training notebook.
Run: streamlit run app.py
"""

import pickle
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ART_DIR = Path(__file__).parent / "artifacts"

st.set_page_config(
    page_title="KSP Crime Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
)

# ----------------------------------------------------------------------
# Data loading (cached)
# ----------------------------------------------------------------------

@st.cache_data
def load_incidents():
    df = pd.read_csv(ART_DIR / "crime_incidents.csv", parse_dates=["date"])
    return df

@st.cache_data
def load_daily_risk():
    return pd.read_csv(ART_DIR / "district_daily_risk.csv", parse_dates=["date"])

@st.cache_data
def load_districts():
    return pd.read_csv(ART_DIR / "districts_reference.csv")

@st.cache_data
def load_hotspots():
    return pd.read_csv(ART_DIR / "hotspot_clusters.csv")

@st.cache_data
def load_graph_tables():
    edges = pd.read_csv(ART_DIR / "graph_edges.csv")
    nodes = pd.read_csv(ART_DIR / "graph_nodes.csv")
    return edges, nodes

@st.cache_data
def load_top_offenders():
    return pd.read_csv(ART_DIR / "top_repeat_offenders.csv")

@st.cache_data
def load_emerging_trends():
    file_path = ART_DIR / "emerging_trend_alerts.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()

@st.cache_data
def load_association_rules():
    file_path = ART_DIR / "association_rules.csv"
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()

# --- NEW DATA LOADERS FOR CHALLENGE REQUIREMENTS ---

@st.cache_data
def load_forecast():
    file_path = ART_DIR / "forecast_30_days.csv"
    if file_path.exists():
        return pd.read_csv(file_path, parse_dates=["date"])
    return pd.DataFrame()

@st.cache_data
def load_correlation():
    file_path = ART_DIR / "correlation_matrix.csv"
    if file_path.exists():
        return pd.read_csv(file_path, index_col=0)
    return pd.DataFrame()

@st.cache_data
def load_json_artifact(filename):
    file_path = ART_DIR / filename
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_resource
def load_risk_model():
    with open(ART_DIR / "risk_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_anomaly_model():
    with open(ART_DIR / "isolation_forest.pkl", "rb") as f:
        return pickle.load(f)

if not ART_DIR.exists():
    st.error(
        "Artifacts folder not found. Run the Kaggle training notebook, download its "
        "`artifacts/` output, and place it next to this app.py file."
    )
    st.stop()

# Load all data
incidents = load_incidents()
daily_risk = load_daily_risk()
districts = load_districts()
hotspots = load_hotspots()
edges, nodes = load_graph_tables()
top_offenders = load_top_offenders()
emerging_alerts = load_emerging_trends()
assoc_rules = load_association_rules()
forecast_df = load_forecast()
correlation_df = load_correlation()
ai_insights = load_json_artifact("ai_insights.json")
kpi_metrics = load_json_artifact("kpi_metrics.json")


# ----------------------------------------------------------------------
# Sidebar navigation + global filters + Report Generator
# ----------------------------------------------------------------------

st.sidebar.title("🛡️ KSP Crime Intelligence")
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Search & AI Query", # <-- NEW PAGE
        "Geospatial Hotspot Map",
        "District Drill-down",
        "Network & Link Analysis",
        "Predictive Risk Dashboard",
        "Anomaly Detection",
        "Trend & Pattern Discovery",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
district_filter = st.sidebar.multiselect(
    "District", sorted(incidents["district"].unique()), default=[]
)
crime_filter = st.sidebar.multiselect(
    "Crime type", sorted(incidents["crime_type"].unique()), default=[]
)
date_range = st.sidebar.date_input(
    "Date range",
    value=(incidents["date"].min().date(), incidents["date"].max().date()),
)

filtered = incidents.copy()
if district_filter:
    filtered = filtered[filtered["district"].isin(district_filter)]
if crime_filter:
    filtered = filtered[filtered["crime_type"].isin(crime_filter)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]

st.sidebar.caption(f"{len(filtered):,} incidents match current filters")

# --- FEATURE 5: REPORT GENERATOR ---
@st.cache_data
def generate_report_csv(df):
    """Generates a downloadable intelligence report in CSV format."""
    summary = df.groupby(['district', 'crime_type']).size().reset_index(name='Total_Cases')
    return summary.to_csv(index=False).encode('utf-8')

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Export Intelligence")
report_csv = generate_report_csv(filtered)
st.sidebar.download_button(
    label="Download Executive Report (CSV)",
    data=report_csv,
    file_name='ksp_executive_intel_report.csv',
    mime='text/csv',
)


# ----------------------------------------------------------------------
# PAGE: Overview (Enhanced with AI Insights & Better KPIs)
# ----------------------------------------------------------------------

if page == "Overview":
    st.title("AI-Driven Crime Analytics & Visualization Platform")
    st.caption("State Crime Records Bureau — Strategic Intelligence Hub")

    # --- FEATURE 7: BETTER KPI CARDS ---
    if kpi_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Today's Crimes", f"{kpi_metrics.get('todays_crimes', 0)}")
        
        growth = kpi_metrics.get('daily_growth_rate', 0)
        c2.metric("Daily Growth Rate", f"{growth}%", delta=f"{growth}%", delta_color="inverse")
        
        c3.metric("Total Cases Analyzed", f"{kpi_metrics.get('total_cases_analyzed', 0):,}")
        c4.metric("Highest Volume District", kpi_metrics.get('highest_volume_district', 'N/A'))
    else:
        # Fallback to original KPIs if JSON is missing
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total incidents", f"{len(filtered):,}")
        c2.metric("Resolution rate", f"{filtered['resolved'].mean()*100:.1f}%")
        c3.metric("Avg severity", f"{filtered['severity_score'].mean():.2f} / 10")
        c4.metric("Anomalous incidents", f"{filtered['is_anomaly'].sum():,}")

    # --- FEATURES 1 & 6: AI INSIGHTS & RECOMMENDATIONS ---
    if ai_insights:
        st.markdown("---")
        st.subheader("💡 Automated AI Insights & Recommendations")
        col_in, col_rec = st.columns(2)
        
        with col_in:
            st.info("**Key Findings:**")
            for insight in ai_insights.get("insights", []):
                st.markdown(f"• {insight}")
                
        with col_rec:
            st.success("**Recommended Actions:**")
            for rec in ai_insights.get("recommendations", []):
                st.markdown(f"• {rec}")
    st.markdown("---")

    # --- EMERGING TREND ALERTS (red-zone spike callouts) ---
    if not emerging_alerts.empty:
        critical = emerging_alerts[emerging_alerts["alert_flag"] == "Critical Spike"].sort_values(
            "z_score", ascending=False
        )
        if not critical.empty:
            st.subheader("🔴 Emerging Trend Alerts")
            st.caption("Crime-type/district combinations whose monthly volume spiked far above their historical average.")
            top_n = critical.head(5)
            cols = st.columns(len(top_n))
            for col, (_, row) in zip(cols, top_n.iterrows()):
                col.markdown(
                    f"""<div style="border:2px solid #d62728;border-radius:8px;padding:10px;
                    background-color:rgba(214,39,40,0.08);animation:pulse 2s infinite;">
                    <b>🔴 {row['crime_type']}</b><br>{row['district']} — {row['year_month']}<br>
                    <span style="color:#d62728;font-weight:bold;">{row['incident_count']} cases
                    (z={row['z_score']:.1f})</span></div>
                    <style>@keyframes pulse {{0%{{opacity:1;}}50%{{opacity:0.55;}}100%{{opacity:1;}}}}</style>
                    """,
                    unsafe_allow_html=True,
                )
            with st.expander(f"View all {len(critical)} critical spikes"):
                st.dataframe(
                    critical[["district", "crime_type", "year_month", "incident_count", "z_score"]],
                    width='stretch',
                )
            st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        by_crime = filtered["crime_type"].value_counts().reset_index()
        by_crime.columns = ["crime_type", "count"]
        fig = px.bar(by_crime, x="count", y="crime_type", orientation="h",
                     title="Incidents by crime type", color="count",
                     color_continuous_scale="Reds")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')
    with col2:
        by_district = filtered["district"].value_counts().head(12).reset_index()
        by_district.columns = ["district", "count"]
        fig = px.bar(by_district, x="count", y="district", orientation="h",
                     title="Top districts by incident count", color="count",
                     color_continuous_scale="Blues")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')

# ----------------------------------------------------------------------
# PAGE: Search & AI Query (NEW)
# ----------------------------------------------------------------------
elif page == "Search & AI Query":
    # --- FEATURES 2 & 4: NATURAL LANGUAGE QUERY & GLOBAL SEARCH ---
    st.title("🔍 Intelligence Search & AI Query")
    st.caption("Ask questions in natural language or search globally across all records.")

    def parse_nlq(query, df):
        """Rule-based engine to handle natural language queries."""
        query = query.lower()
        temp_df = df.copy()
        
        districts = df['district'].astype(str).str.lower().unique()
        detected_district = next((d for d in districts if d in query), None)
        if detected_district:
            temp_df = temp_df[temp_df['district'].astype(str).str.lower() == detected_district]
            
        crime_heads = df['crime_type'].astype(str).str.lower().unique()
        detected_crime = next((c for c in crime_heads if c in query), None)
        if detected_crime:
            temp_df = temp_df[temp_df['crime_type'].astype(str).str.lower() == detected_crime]

        if "highest" in query or "maximum" in query:
            return temp_df.groupby('district').size().sort_values(ascending=False).head(1).reset_index(name="Incident Count")
            
        return temp_df

    search_mode = st.radio("Search Mode", ["Natural Language Query", "Global Search (Exact Match)"], horizontal=True)
    user_query = st.text_input("Ask a question (e.g., 'Show kidnapping cases in Mysore') or search an ID (e.g., 'INC-000001'):")

    if user_query:
        if search_mode == "Natural Language Query":
            result_df = parse_nlq(user_query, incidents)
            st.success(f"Found {len(result_df)} records matching your query intent.")
            st.dataframe(result_df, width='stretch')
        else:
            # Global Search across all columns
            mask = incidents.apply(lambda row: row.astype(str).str.contains(user_query, case=False).any(), axis=1)
            result_df = incidents[mask]
            st.success(f"Found {len(result_df)} exact matches.")
            st.dataframe(result_df, width='stretch')


# ----------------------------------------------------------------------
# PAGE: Geospatial Hotspot Map
# ----------------------------------------------------------------------
elif page == "Geospatial Hotspot Map":
    st.title("Geospatial Crime Hotspot Map")
    st.caption("Replaces static Excel sheets with an interactive, spatial view of crime clusters.")

    map_mode = st.radio("Map layer", ["Individual incidents", "Hotspot clusters"], horizontal=True)

    if map_mode == "Individual incidents":
        sample = filtered.sample(min(4000, len(filtered)), random_state=1) if len(filtered) > 4000 else filtered
        fig = px.scatter_map(
            sample, lat="latitude", lon="longitude", color="crime_type",
            hover_data=["district", "date", "severity_score"],
            zoom=6, height=650, map_style="carto-positron",
        )
        st.plotly_chart(fig, width='stretch')
        if len(filtered) > 4000:
            st.caption("Showing a 4,000-point sample for render performance.")
    else:
        fig = px.scatter_map(
            hotspots, lat="center_lat", lon="center_lon", size="size",
            color="dominant_crime", hover_data=["avg_hour", "size"],
            zoom=6, height=650, map_style="carto-positron",
            title="Spatiotemporal hotspot clusters (DBSCAN)",
        )
        st.plotly_chart(fig, width='stretch')
        st.dataframe(hotspots.sort_values("size", ascending=False), width='stretch')

# ----------------------------------------------------------------------
# PAGE: District Drill-down
# ----------------------------------------------------------------------
elif page == "District Drill-down":
    st.title("District-Level Drill-down")
    dist = st.selectbox("Select district", sorted(incidents["district"].unique()))
    d_data = incidents[incidents["district"] == dist]

    c1, c2, c3 = st.columns(3)
    c1.metric("Incidents", f"{len(d_data):,}")
    c2.metric("Resolution rate", f"{d_data['resolved'].mean()*100:.1f}%")
    c3.metric("Avg severity", f"{d_data['severity_score'].mean():.2f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(d_data, names="crime_type", title=f"Crime mix — {dist}", hole=0.4)
        st.plotly_chart(fig, width='stretch')
    with col2:
        heat = d_data.groupby(["day_of_week", "hour"]).size().reset_index(name="count")
        fig = px.density_heatmap(heat, x="hour", y="day_of_week", z="count",
                                  title="When crime happens (hour vs weekday)", color_continuous_scale="Inferno")
        st.plotly_chart(fig, width='stretch')

# ----------------------------------------------------------------------
# PAGE: Network & Link Analysis
# ----------------------------------------------------------------------
elif page == "Network & Link Analysis":
    st.title("Criminological Network & Link Analysis")
    st.caption("Visually connects suspects, victims, and locations to reveal organized crime structures.")

    # --- BETTER NETWORK GRAPH (Pyvis Interactive) ---
    # NOTE: the full statewide graph has ~28,000 nodes / ~39,000 edges. Rendering that many
    # nodes with physics enabled in a browser tab will hang/freeze it, regardless of machine —
    # this is not a data problem, it's a "too much to force-layout in a DOM" problem.
    # Fix: build a capped subgraph (top-N highest-degree suspects + their direct connections)
    # on the fly, sized so it always stays render-safe.
    st.subheader("🕸️ Interactive Association Graph (top connected suspects)")

    @st.cache_data(show_spinner=False)
    def build_network_html(top_n: int, edges_df: pd.DataFrame, nodes_df: pd.DataFrame) -> str:
        node_type_map = dict(zip(nodes_df["node"], nodes_df["type"]))
        Gfull = nx.from_pandas_edgelist(edges_df, source="source", target="target")

        suspects_all = [n for n in Gfull.nodes() if node_type_map.get(n) == "suspect"]
        degree_map_all = {n: Gfull.degree(n) for n in suspects_all}
        top_suspects = {n for n, _ in sorted(degree_map_all.items(), key=lambda x: x[1], reverse=True)[:top_n]}

        keep_nodes = set(top_suspects)
        for s in top_suspects:
            keep_nodes.update(Gfull.neighbors(s))
        Hsub = Gfull.subgraph(keep_nodes)

        from pyvis.network import Network
        net = Network(height="600px", width="100%", bgcolor="#1E1E1E", font_color="white", notebook=False, cdn_resources="in_line")
        color_map = {"suspect": "#e74c3c", "victim": "#3498db", "location": "#2ecc71"}
        for node in Hsub.nodes():
            n_type = node_type_map.get(node, "unknown")
            net.add_node(node, label=str(node), title=f"Type: {n_type}", color=color_map.get(n_type, "#888"))
        for u, v in Hsub.edges():
            net.add_edge(u, v)

        # Physics on for a real force-directed layout, but with stabilization capped so it
        # settles quickly and doesn't churn indefinitely on a large-ish subgraph.
        net.set_options("""
        {
          "physics": {
            "stabilization": {"enabled": true, "iterations": 150},
            "barnesHut": {"gravitationalConstant": -12000, "springLength": 120}
          }
        }
        """)
        # Get the HTML directly from pyvis rather than writing to disk: net.save_graph()/
        # write_html() opens the file with the OS default encoding, which is cp1252 on
        # Windows and can't encode all the characters in the embedded vis-network JS,
        # causing UnicodeEncodeError. generate_html() just renders the template in memory.
        html = net.generate_html(notebook=False)
        return html, Hsub.number_of_nodes(), Hsub.number_of_edges()

    top_n_suspects = st.slider(
        "Number of top-connected suspects to include", min_value=20, max_value=300, value=100, step=20,
        help="Limits the graph to the N highest-degree suspects plus their direct victim/location connections. "
             "Kept capped so the browser doesn't have to force-layout the full ~28,000-node statewide graph, "
             "which is what caused the page to hang.",
    )
    with st.spinner("Building network graph…"):
        html_data, shown_nodes, shown_edges = build_network_html(top_n_suspects, edges, nodes)
    st.caption(f"Showing {shown_nodes:,} nodes and {shown_edges:,} edges (capped for render performance).")
    components.html(html_data, height=650, scrolling=True)

    st.markdown("---")
    st.subheader("Ego Network (Suspect Deep-Dive)")
    st.caption("Static fallback and specific suspect drill-down.")
    
    suspect_id = st.selectbox("Search specific Suspect ID", top_offenders["suspect_id"].tolist())

    # build ego graph
    sub_edges = edges[(edges["source"] == suspect_id) | (edges["target"] == suspect_id)]
    ego_nodes = set(sub_edges["source"]).union(set(sub_edges["target"]))
    node_type = dict(zip(nodes["node"], nodes["type"]))

    G = nx.Graph()
    for _, row in sub_edges.iterrows():
        G.add_edge(row["source"], row["target"])
    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), mode="lines", hoverinfo="none")

    color_map = {"suspect": "#d62728", "victim": "#1f77b4", "location": "#2ca02c"}
    node_x, node_y, node_color, node_text = [], [], [], []
    for n in G.nodes():
        node_x.append(pos[n][0]); node_y.append(pos[n][1])
        t = node_type.get(n, "unknown")
        node_color.append(color_map.get(t, "#888"))
        node_text.append(f"{n} ({t})")

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center",
        marker=dict(size=14, color=node_color, line=dict(width=1, color="white")), hoverinfo="text",
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, height=500, title=f"Ego network — {suspect_id}",
                       xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, width='stretch')

    # --- ASSOCIATION DETECTION (hidden Act/Crime associations) ---
    st.markdown("---")
    st.subheader("🔗 Association Detection")
    st.caption("Statistically significant Act ↔ Crime-type associations mined from case records — "
               "relationships that are easy to miss in isolated Excel sheets.")
    if not assoc_rules.empty:
        top_rules = assoc_rules.sort_values("lift", ascending=False).head(15).copy()
        top_rules["antecedents"] = top_rules["antecedents"].astype(str)
        top_rules["consequents"] = top_rules["consequents"].astype(str)
        display_cols = [c for c in ["antecedents", "consequents", "support", "confidence", "lift"]
                         if c in top_rules.columns]
        st.dataframe(
            top_rules[display_cols].style.format(
                {"support": "{:.3f}", "confidence": "{:.2f}", "lift": "{:.2f}"}
            ),
            width='stretch',
        )
        best = top_rules.iloc[0]
        st.info(
            f"Strongest association: **{best['antecedents']} → {best['consequents']}** "
            f"(confidence {best['confidence']*100:.0f}%, lift {best['lift']:.1f}×) — "
            "meaning cases with the first attribute are far more likely to also involve the second "
            "than random chance would predict."
        )
    else:
        st.warning("No association-rule artifact found. Run the association-rule mining cell in the training notebook first.")

# ----------------------------------------------------------------------
# PAGE: Predictive Risk Dashboard
# ----------------------------------------------------------------------
elif page == "Predictive Risk Dashboard":
    st.title("Predictive Risk Scoring & Explainable AI")
    st.caption("AI-driven forecast of which district-days are likely to be high-risk.")

    risk_bundle = load_risk_model()
    model = risk_bundle["model"]
    le_district = risk_bundle["district_encoder"]
    feature_cols = risk_bundle["features"]

    latest = daily_risk.sort_values("date").groupby("district").tail(1).copy()
    latest["district_enc"] = le_district.transform(latest["district"])
    X_latest = latest[feature_cols].fillna(0)
    latest["risk_probability"] = model.predict_proba(X_latest)[:, 1]
    latest = latest.sort_values("risk_probability", ascending=False).reset_index(drop=True)

    # Risk tier thresholds — used consistently everywhere below
    def risk_tier(p):
        if p >= 0.70:
            return "High Risk", "🔴"
        elif p >= 0.40:
            return "Moderate Risk", "🟠"
        return "Low Risk", "🟢"

    latest["risk_tier"] = latest["risk_probability"].apply(lambda p: risk_tier(p)[0])

    # Human-readable feature labels + direction wording, used for every district (not hardcoded to one)
    FEATURE_LABELS = {
        "urbanization": "Urbanization level",
        "pop_density": "Population density",
        "literacy": "Literacy rate",
        "unemployment": "Unemployment rate",
        "rolling_7d_count": "Recent 7-day incident trend",
        "day_of_week": "Day-of-week pattern",
        "month": "Seasonal (month) pattern",
        "avg_severity": "Average case severity",
        "weapon_rate": "Weapon-involvement rate",
        "district_enc": "District identity baseline",
    }
    # Whether a HIGHER value of this feature is intuitively bad (raises risk) or good (lowers risk)
    RAISES_RISK_WHEN_HIGH = {
        "urbanization": True, "pop_density": True, "unemployment": True,
        "rolling_7d_count": True, "avg_severity": True, "weapon_rate": True,
        "literacy": False, "day_of_week": None, "month": None, "district_enc": None,
    }

    # Baseline stats (mean/std across all district-days) so we can say whether a value is
    # unusually high/low for a given district, not just its raw magnitude
    baseline = daily_risk[feature_cols].fillna(0)
    feat_mean = baseline.mean()
    feat_std = baseline.std().replace(0, 1)
    importances = pd.Series(model.feature_importances_, index=feature_cols)

    def explain_row(row):
        """Rule-based, per-instance explanation: for each feature, combine
        (a) how much the model globally relies on it (feature_importances_) with
        (b) how unusual this district's value is vs the statewide baseline (z-score).
        This is not a true SHAP value, but it's a reasonable, fast, fully-transparent
        stand-in for a prototype: features the model cares about AND that are unusual
        for this district rise to the top."""
        contributions = []
        for feat in feature_cols:
            z = (row[feat] - feat_mean[feat]) / feat_std[feat]
            weight = importances.get(feat, 0)
            signed_score = z * weight
            raises_high = RAISES_RISK_WHEN_HIGH.get(feat)
            if raises_high is False:
                signed_score = -signed_score  # flip so positive score always means "pushes risk up"
            contributions.append({
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "value": row[feat],
                "z_score": z,
                "score": signed_score,
            })
        contrib_df = pd.DataFrame(contributions)
        return contrib_df.reindex(contrib_df["score"].abs().sort_values(ascending=False).index)

    st.markdown("### 🧠 Model Explainability (XAI)")
    st.caption(
        "Rule-based explanation (prototype-grade, not a true SHAP/LIME value): for each district we combine "
        "how much the model relies on a feature overall with how unusual that district's value is right now."
    )

    district_options = latest["district"].tolist()
    selected_district = st.selectbox(
        "Select a district to explain", district_options,
        index=0,  # defaults to the highest-risk district, since `latest` is sorted descending
    )
    row = latest[latest["district"] == selected_district].iloc[0]
    tier_label, tier_icon = risk_tier(row["risk_probability"])

    col_score, col_reasons = st.columns([1, 2])
    with col_score:
        prob = row["risk_probability"]
        st.metric(
            "Risk Confidence Score", f"{prob * 100:.1f}%",
            delta=tier_label, delta_color="inverse" if tier_label != "Low Risk" else "normal",
        )
        st.caption(f"{tier_icon} Classified as **{tier_label}** (High ≥ 70%, Moderate ≥ 40%, else Low).")

    with col_reasons:
        st.write(f"**Why '{selected_district}' was classified as {tier_label}:**")
        top_factors = explain_row(row).head(4)
        any_shown = False
        for _, f in top_factors.iterrows():
            if abs(f["z_score"]) < 0.15:
                continue  # value is basically at the statewide average — not a meaningful driver
            any_shown = True
            direction_icon = "🔴 **+Raises risk**" if f["score"] > 0 else "🟢 **-Lowers risk**"
            unusualness = "much higher than" if f["z_score"] > 0 else "much lower than"
            st.write(
                f"{direction_icon} **{f['label']}** is {unusualness} the statewide average "
                f"(value {f['value']:.2f}, z={f['z_score']:+.2f})."
            )
        if not any_shown:
            st.write("🟡 No single feature is far from the statewide average — risk here tracks close to baseline.")

    st.markdown("---")

    fig = px.bar(
        latest, x="risk_probability", y="district", orientation="h",
        color="risk_probability", color_continuous_scale="RdYlGn_r",
        title="Predicted high-risk probability by district (most recent snapshot)",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width='stretch')

    with st.expander("📋 All districts — risk tier & confidence score"):
        st.dataframe(
            latest[["district", "risk_probability", "risk_tier"]]
            .rename(columns={"risk_probability": "confidence_score"})
            .style.format({"confidence_score": "{:.1%}"}),
            width='stretch',
        )

    st.subheader("Feature Importance Analysis (global — across all districts)")
    global_importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    global_importances.index = [FEATURE_LABELS.get(f, f) for f in global_importances.index]
    fig2 = px.bar(global_importances, orientation="h", title="What drives the risk score, on average")
    st.plotly_chart(fig2, width='stretch')

# ----------------------------------------------------------------------
# PAGE: Anomaly Detection
# ----------------------------------------------------------------------
elif page == "Anomaly Detection":
    st.title("Anomaly Detection")
    st.caption("Incidents that deviate from standard behavioral patterns — for linking complex or unusual cases.")

    anomalies = filtered[filtered["is_anomaly"]]
    st.metric("Anomalous incidents in current filter", f"{len(anomalies):,}")

    fig = px.scatter_map(
        anomalies, lat="latitude", lon="longitude", color="crime_type",
        size="severity_score", hover_data=["district", "date", "hour"],
        zoom=6, height=600, map_style="carto-positron", title="Anomalous incidents map",
    )
    st.plotly_chart(fig, width='stretch')

# ----------------------------------------------------------------------
# PAGE: Trend & Pattern Discovery
# ----------------------------------------------------------------------
elif page == "Trend & Pattern Discovery":
    st.title("Pattern & Trend Discovery")

    # --- FEATURE 10: FORECASTING ---
    if not forecast_df.empty:
        st.subheader("📈 30-Day Crime Volume Forecast")
        fig_forecast = go.Figure()
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'], y=forecast_df['Forecast'],
            mode='lines', name='Forecasted Volume', line=dict(color='orange', width=3)
        ))
        
        # Add Confidence Intervals
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'], y=forecast_df['Upper_CI'],
            mode='lines', marker=dict(color="#444"), line=dict(width=0),
            showlegend=False, name='Upper Bound'
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_df['date'], y=forecast_df['Lower_CI'],
            mode='lines', marker=dict(color="#444"), line=dict(width=0),
            fillcolor='rgba(255, 165, 0, 0.2)', fill='tonexty',
            showlegend=False, name='Lower Bound'
        ))
        
        fig_forecast.update_layout(title="Statewide Expected Incident Volume (Next 30 Days)", hovermode="x")
        st.plotly_chart(fig_forecast, width='stretch')
        st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        crime_sel = st.selectbox("Crime type", sorted(incidents["crime_type"].unique()))
        trend = incidents[incidents["crime_type"] == crime_sel].groupby(
            incidents["date"].dt.to_period("M")
        ).size().reset_index(name="count")
        trend["date"] = trend["date"].dt.to_timestamp()
        fig = px.line(trend, x="date", y="count", title=f"{crime_sel} trend over time", markers=True)
        st.plotly_chart(fig, width='stretch')
    with col2:
        mo_counts = incidents[incidents["crime_type"] == crime_sel]["mo"].value_counts().reset_index()
        mo_counts.columns = ["modus_operandi", "count"]
        fig = px.bar(mo_counts, x="count", y="modus_operandi", orientation="h",
                     title=f"Modus Operandi breakdown — {crime_sel}")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    
    # --- FEATURE 9: CORRELATION ANALYSIS ---
    if not correlation_df.empty:
        st.subheader("🧩 Socio-Economic Correlation Matrix")
        st.caption("Identify relationships between environmental factors and crime severity.")
        
        fig_corr = px.imshow(
            correlation_df, text_auto=True, aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, width='stretch')

    # --- EMERGING TREND ALERTS (full table view) ---
    if not emerging_alerts.empty:
        st.markdown("---")
        st.subheader("🚨 Emerging Trend Alert Log")
        st.caption("All district/crime-type combinations flagged for statistically significant month-over-month spikes.")
        flag_filter = st.multiselect(
            "Filter by alert level",
            sorted(emerging_alerts["alert_flag"].unique()),
            default=list(emerging_alerts["alert_flag"].unique()),
        )
        alerts_view = emerging_alerts[emerging_alerts["alert_flag"].isin(flag_filter)].sort_values(
            "z_score", ascending=False
        )
        st.dataframe(
            alerts_view[["district", "crime_type", "year_month", "incident_count", "z_score", "alert_flag"]],
            width='stretch',
        )