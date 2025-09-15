# takamaka_dashboard.py
import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors  # HEX -> RGBA

# =========================
# Config
# =========================
FILENAME   = "one.state"   # cambia se serve
OUTPUT_DIR = "output"
MAX_SLOT   = 2399

# Address -> readable label
LABEL = {
    "Be2ntZsJLQ0Kmp_KRv-281k-x5pWqnuUNNzFqsi35lQ.": "main_1",
    "_0OHvruLBmlRlDnCEmKdxyc65hBynAxRtiuQ8BHkrkM.": "overflow_1",
    "TgLaccJovUQPjgNcU1ZsS-tYrYnuNTRtjLdVvAqg3kI.": "main_2",
    "83MqfFUtbX6B1pgEYc71Sa7hfwoVmd3KDw7pawj3YDY.": "overflow_2",
    "Lb5sLN6alt5KcdMNE74onoOuQL8Y0eYgGZKqDOAUSTU.": "main_3",
    "y4V32jfDh8MyUviBvbzi_dMT54N_IqVxF8HtdWBfKiA.": "overflow_3",
}

# Canonical orders (so nodes always appear, even if 0)
MAIN_ADDRS = [
    "Be2ntZsJLQ0Kmp_KRv-281k-x5pWqnuUNNzFqsi35lQ.",
    "TgLaccJovUQPjgNcU1ZsS-tYrYnuNTRtjLdVvAqg3kI.",
    "Lb5sLN6alt5KcdMNE74onoOuQL8Y0eYgGZKqDOAUSTU.",
]
OVERFLOW_ADDRS = [
    "_0OHvruLBmlRlDnCEmKdxyc65hBynAxRtiuQ8BHkrkM.",
    "83MqfFUtbX6B1pgEYc71Sa7hfwoVmd3KDw7pawj3YDY.",
    "y4V32jfDh8MyUviBvbzi_dMT54N_IqVxF8HtdWBfKiA.",
]

# Colors
PALETTE = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2", "#EDC948"]
BAR_TRSP = ["rgba(78,121,167,0.60)", "rgba(242,142,43,0.60)", "rgba(89,161,79,0.60)"]

# ===== Stake static values (example @ epoch=2, slot=800) =====
STAKE_EPOCH = 1
STAKE_SLOT  = 799
# Put stake for mains here; missing mains default to 0
STAKE_RAW = {
    "Be2ntZsJLQ0Kmp_KRv-281k-x5pWqnuUNNzFqsi35lQ.": 98_999_369_600_301_770,
    "TgLaccJovUQPjgNcU1ZsS-tYrYnuNTRtjLdVvAqg3kI.": 0,
    "Lb5sLN6alt5KcdMNE74onoOuQL8Y0eYgGZKqDOAUSTU.": 0,
}

# =========================
# Helpers
# =========================
def hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    r, g, b = [int(c * 255) for c in mcolors.to_rgb(hex_color)]
    return f"rgba({r},{g},{b},{alpha})"

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================
# Stake (mains only; missing -> 0)
# =========================
def build_stake_figure() -> go.Figure:
    names  = [LABEL[a] for a in MAIN_ADDRS]
    values = [int(STAKE_RAW.get(a, 0)) for a in MAIN_ADDRS]
    total  = max(sum(values), 1)
    labels = [f"{v:,.0f} ({v/total:.1%})" for v in values]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(color=BAR_TRSP, line=dict(width=1, color="black")),
        text=labels, textposition="outside",
        hovertemplate="%{x}<br>Stake: %{y:,}<extra></extra>",
        showlegend=False,
        name="Stake",
    ))
    fig.update_layout(
        title=f"Stake per node (epoch {STAKE_EPOCH}, slot {STAKE_SLOT})",
        xaxis_title="Node",
        yaxis_title="Stake (TKG)",
        template="plotly_white",
        margin=dict(l=80, r=80, t=80, b=80),
    )
    ymax = max(values) if any(values) else 1
    fig.update_yaxes(range=[0, ymax * 1.15], tickformat=",")
    return fig

# =========================
# Penalty slots (mains only; missing -> 0)
# =========================
def build_penalty_figure(state: dict) -> go.Figure:
    epoch_str = str(state.get("epoch", 0))
    op_epoch  = state.get("operationalRecord", {}).get(epoch_str, {})
    mains     = op_epoch.get("urlMainRecords", {}) or {}

    values_by_addr = {a: int(mains.get(a, {}).get("penaltySlots", 0)) for a in MAIN_ADDRS}
    names  = [LABEL[a] for a in MAIN_ADDRS]
    values = [values_by_addr[a] for a in MAIN_ADDRS]
    total  = max(sum(values), 1)
    colors = [hex_to_rgba(PALETTE[i], 0.6) for i in range(3)]
    text   = [f"{v:,.0f} ({v/total:.1%})" for v in values]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(color=colors, line=dict(width=1, color="black")),
        text=text, textposition="outside",
        hovertemplate="%{x}<br>Penalty slots: %{y:,.0f}<br>Share: %{customdata:.1%}<extra></extra>",
        customdata=[v/total for v in values],
        showlegend=False,
    ))
    fig.update_layout(
        title=f"Penalty slots per node (epoch {epoch_str})",
        xaxis_title="Node",
        yaxis_title="Penalty slots",
        template="plotly_white",
        margin=dict(l=80, r=80, t=80, b=80),
    )
    ymax = max(values) if any(values) else 1
    fig.update_yaxes(range=[0, ymax * 1.15], tickformat=",")
    return fig

# =========================
# Summary (Delivered + Validated %) — overflows only; missing -> 0
# =========================
def build_summary_figure(state: dict) -> go.Figure:
    epoch_str = str(state.get("epoch", 0))

    # --- Delivered blocks (overflows) ---
    op_epoch  = state.get("operationalRecord", {}).get(epoch_str, {})
    overflows = op_epoch.get("urlOverflowCounter", {}) or {}
    bar_vals  = [int(overflows.get(a, {}).get("deliveredBlocks", 0)) for a in OVERFLOW_ADDRS]
    bar_names = [LABEL[a] for a in OVERFLOW_ADDRS]
    total_deliv = max(sum(bar_vals), 1)
    perc_text   = [f"{v:,.0f} ({v/total_deliv:.1%})" for v in bar_vals]
    bar_colors  = [hex_to_rgba(PALETTE[i], 0.6) for i in range(3)]

    # --- Validated slots cumulative share (overflows) ---
    addr_to_slots = state.get("currentEpochSlotDistribution", {}) or {}
    T = MAX_SLOT + 1
    x = np.arange(T)

    # Build indicators only for overflow list (zeros if missing)
    indicators = []
    for addr in OVERFLOW_ADDRS:
        slots = [s for s in addr_to_slots.get(addr, []) if 0 <= s <= MAX_SLOT]
        v = np.zeros(T, dtype=int)
        if slots:
            v[slots] = 1
        indicators.append(v)
    indicators = np.vstack(indicators)  # shape (3, T)

    cum_by_node = indicators.cumsum(axis=1)        # (3, T)
    cum_total   = cum_by_node.sum(axis=0)
    cum_total   = np.where(cum_total == 0, 1, cum_total)  # avoid /0
    shares      = cum_by_node / cum_total

    # --- Subplots ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Delivered blocks per node (epoch {epoch_str})",
                        f"Validated slots in percentage (epoch {epoch_str})"),
        column_widths=[0.33, 0.67]
    )

    # Bars (left)
    fig.add_trace(
        go.Bar(
            x=bar_names, y=bar_vals,
            marker=dict(color=bar_colors, line=dict(width=1, color="black")),
            text=perc_text, textposition="outside",
            hovertemplate="%{x}<br>Blocks: %{y:,.0f}<br>Share: %{customdata:.1%}<extra></extra>",
            customdata=[v/total_deliv for v in bar_vals],
            showlegend=False,
        ),
        row=1, col=1
    )

    # Area (right)
    for i, addr in enumerate(OVERFLOW_ADDRS):
        name = LABEL[addr]
        line_color = PALETTE[i]
        fill_color = hex_to_rgba(line_color, 0.5)
        fig.add_trace(
            go.Scatter(
                x=x, y=shares[i],
                name=name, mode="lines",
                line=dict(color=line_color, width=2),
                stackgroup="one",
                fill="tonexty", fillcolor=fill_color,
                hovertemplate="Slot=%{x}<br>" + f"{name} cumulative share=%{{y:.1%}}<extra></extra>",
            ),
            row=1, col=2
        )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Node",
        margin=dict(l=60, r=40, t=70, b=50),
    )
    ymax = max(bar_vals) if any(bar_vals) else 1
    fig.update_yaxes(range=[0, ymax * 1.15], tickformat=",", row=1, col=1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=2)
    fig.update_xaxes(range=[0, MAX_SLOT], row=1, col=2)
    return fig

# =========================
# Run
# =========================
if __name__ == "__main__":
    ensure_outdir(OUTPUT_DIR)
    state = load_state(FILENAME)
    epoch_str = str(state.get("epoch", 0))

    # 1) Stake (mains only)
    stake_fig = build_stake_figure()
    stake_png = os.path.join(OUTPUT_DIR, f"stake_epoch_{STAKE_EPOCH}_slot_{STAKE_SLOT}.png")
    stake_fig.write_image(stake_png, width=1200, height=600, scale=2)
    print(f"[OK] Saved: {stake_png}")

    # 2) Penalty slots (mains only)
    penalty_fig = build_penalty_figure(state)
    penalty_png = os.path.join(OUTPUT_DIR, f"penalty_slots_epoch_{epoch_str}.png")
    penalty_fig.write_image(penalty_png, width=1000, height=600, scale=2)
    print(f"[OK] Saved: {penalty_png}")

    # 3) Summary (overflows only)
    summary_fig = build_summary_figure(state)
    summary_png = os.path.join(OUTPUT_DIR, f"summary_epoch_{epoch_str}.png")
    summary_fig.write_image(summary_png, width=1600, height=600, scale=2)
    print(f"[OK] Saved: {summary_png}")
