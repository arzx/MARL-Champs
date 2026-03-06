import json
import os
import time
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_state.json")
REFRESH_INTERVAL = 10  # seconds

st.set_page_config(
    page_title="MARL-Champs Live Dashboard",
    layout="wide",
    page_icon="📈",
)

st.title("MARL-Champs — Live Training Dashboard")

if not Path(STATE_FILE).exists():
    st.info("Waiting for training to start...")
    time.sleep(REFRESH_INTERVAL)
    st.rerun()

with open(STATE_FILE) as f:
    state = json.load(f)

iteration = state["iteration"]
agents = state["agents"]

st.caption(f"Last update: iteration {iteration} — auto-refreshing every {REFRESH_INTERVAL}s")

# --- Leaderboard ---
st.subheader("Leaderboard")

sorted_agents = sorted(
    agents.items(),
    key=lambda x: x[1]["episode_return_mean"],
    reverse=True,
)

medals = {0: "🥇", 1: "🥈", 2: "🥉"}
cols = st.columns(len(sorted_agents))
for rank, (agent_id, data) in enumerate(sorted_agents):
    with cols[rank]:
        name = data.get("name", agent_id)
        avatar_url = data.get("avatar_url", "")
        prev = data.get("prev_episode_return_mean")
        delta = f"{data['episode_return_mean'] - prev:.6f}" if prev is not None else None
        if avatar_url:
            st.markdown(
                f'<img src="{avatar_url}" style="border-radius:8px;display:block;margin:0 auto 6px auto">',
                unsafe_allow_html=True,
            )
        st.metric(
            label=f"{medals.get(rank, str(rank + 1))} {name}",
            value=f"{data['episode_return_mean']:.6f}",
            delta=delta,
        )

st.divider()

# --- Return history chart ---
histories = {
    agents[aid].get("name", aid): agents[aid].get("history", [])
    for aid in agents
    if agents[aid].get("history")
}
if histories:
    st.subheader("Episode Return History")
    fig = go.Figure()
    for label, history in histories.items():
        fig.add_trace(go.Scatter(y=history, name=label, mode="lines"))
    fig.update_layout(
        xaxis_title="Update (every shap_interval iterations)",
        yaxis_title="Episode Return Mean",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

# --- SHAP strategy charts ---
st.subheader("Agent Strategies — Feature Importance (SHAP on BUY action)")
st.caption("Which market features most influence each agent's decision to buy")

cols = st.columns(len(sorted_agents))
for i, (agent_id, data) in enumerate(sorted_agents):
    with cols[i]:
        name = data.get("name", agent_id)
        st.markdown(f"**{name}**")
        top_features = data.get("top_features", [])
        if top_features:
            names = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            fig = go.Figure(go.Bar(
                x=values[::-1],
                y=names[::-1],
                orientation="h",
                marker_color="steelblue",
            ))
            fig.update_layout(
                height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Mean |SHAP|",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Computing SHAP...")

time.sleep(REFRESH_INTERVAL)
st.rerun()
