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
FILENAME   = "three.state"
OUTPUT_DIR = "output"
MAX_SLOT   = 2399

# Map indirizzo -> etichetta leggibile
LABEL = {
    "Be2ntZsJLQ0Kmp_KRv-281k-x5pWqnuUNNzFqsi35lQ.": "main_1",
    "_0OHvruLBmlRlDnCEmKdxyc65hBynAxRtiuQ8BHkrkM.": "overflow_1",
    "TgLaccJovUQPjgNcU1ZsS-tYrYnuNTRtjLdVvAqg3kI.": "main_2",
    "83MqfFUtbX6B1pgEYc71Sa7hfwoVmd3KDw7pawj3YDY.": "overflow_2",
    "Lb5sLN6alt5KcdMNE74onoOuQL8Y0eYgGZKqDOAUSTU.": "main_3",
    "y4V32jfDh8MyUviBvbzi_dMT54N_IqVxF8HtdWBfKiA.": "overflow_3",
}

# Palette (Tableau 10)
PALETTE = [
    "#4E79A7", "#F28E2B", "#59A14F",
    "#E15759", "#76B7B2", "#EDC948",
    "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]

# Palette trasparente stile penalty/stake
BAR_TRSP = [
    "rgba(78,121,167,0.60)",  # blu
    "rgba(242,142,43,0.60)",  # arancione
    "rgba(89,161,79,0.60)",   # verde
]

# === Stake statico (epoch=2, slot=800) ===
STAKE_EPOCH = 2
STAKE_SLOT  = 800
# chiave = indirizzo main, valore = stake
STAKE_RAW = {
    "Be2ntZsJLQ0Kmp_KRv-281k-x5pWqnuUNNzFqsi35lQ.": 39_576_467_680_234_554,
    "Lb5sLN6alt5KcdMNE74onoOuQL8Y0eYgGZKqDOAUSTU.": 29_711_022_924_933_334,
    "TgLaccJovUQPjgNcU1ZsS-tYrYnuNTRtjLdVvAqg3kI.": 29_711_922_699_333_335,
}

# =========================
# Helpers
# =========================
def hex_to_rgba(hex_color: str, alpha: float = 0.6) -> str:
    r, g, b = [int(c * 255) for c in mcolors.to_rgb(hex_color)]
    return f"rgba({r},{g},{b},{alpha})"

def load_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# =========================
# Figure: Stake (solo green)
# =========================
def build_stake_figure() -> go.Figure:
    # converti indirizzi → nomi leggibili
    items = [(LABEL.get(addr, addr), val) for addr, val in STAKE_RAW.items()]
    # ordina per valore decrescente per estetica
    items.sort(key=lambda kv: kv[1], reverse=True)

    names  = [n for n, _ in items]
    values = [v for _, v in items]
    total  = sum(values) if values else 1
    labels = [f"{v:,.0f} ({v/total:.1%})" for v in values]

    fig = go.Figure(go.Bar(
        x=names, y=values,
        marker=dict(color=BAR_TRSP[:len(values)], line=dict(width=1, color="black")),
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
    if values:
        fig.update_yaxes(range=[0, max(values) * 1.15], tickformat=",")
    return fig

# =========================
# Figure: Summary (Delivered + Validated %)
# =========================
def build_combined_figure(state: dict) -> go.Figure:
    epoch_str = str(state.get("epoch", 0))

    # Area chart data (validated cumulative share)
    addr_to_slots = state.get("currentEpochSlotDistribution", {})
    addresses = sorted(addr_to_slots.keys(), key=lambda k: LABEL.get(k, k))

    T = MAX_SLOT + 1
    x = np.arange(T)

    indicators = []
    for addr in addresses:
        slots = [s for s in addr_to_slots.get(addr, []) if 0 <= s <= MAX_SLOT]
        v = np.zeros(T, dtype=int)
        if slots:
            v[slots] = 1
        indicators.append(v)
    indicators = np.vstack(indicators) if indicators else np.zeros((0, T), dtype=int)

    cum_by_node = indicators.cumsum(axis=1)
    cum_total   = cum_by_node.sum(axis=0)
    cum_total   = np.where(cum_total == 0, 1, cum_total)
    shares = cum_by_node / cum_total

    # Bar chart data (delivered blocks)
    op_epoch  = state.get("operationalRecord", {}).get(epoch_str, {})
    overflows = op_epoch.get("urlOverflowCounter", {})
    delivered = {addr: meta.get("deliveredBlocks", 0) for addr, meta in overflows.items()}
    items = sorted(delivered.items(), key=lambda kv: kv[1], reverse=True)
    bar_addresses = [addr for addr, _ in items]
    bar_values    = [v for _, v in items]
    bar_names     = [LABEL.get(a, a) for a in bar_addresses]
    total_deliv   = sum(bar_values) if bar_values else 1
    perc_text     = [f"{v:,.0f} ({v/total_deliv:.1%})" for v in bar_values]
    bar_colors    = [hex_to_rgba(PALETTE[i % len(PALETTE)], alpha=0.6) for i in range(len(bar_values))]

    # Subplots: bars left (1/3), area right (2/3)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Delivered blocks per node (epoch {epoch_str})",
                        f"Validated slots in percentage (epoch {epoch_str})"),
        column_widths=[0.33, 0.67]
    )

    # Bars (left)
    fig.add_trace(
        go.Bar(
            x=bar_names, y=bar_values,
            marker=dict(color=bar_colors, line=dict(width=1, color="black")),
            text=perc_text, textposition="outside",
            hovertemplate="%{x}<br>Blocks: %{y:,.0f}<br>Share: %{customdata:.1%}<extra></extra>",
            customdata=[v/total_deliv for v in bar_values],
            showlegend=False
        ),
        row=1, col=1
    )

    # Area (right, 100% stacked)
    for i, addr in enumerate(addresses):
        name = LABEL.get(addr, addr)
        line_color = PALETTE[i % len(PALETTE)]
        fill_color = hex_to_rgba(line_color, alpha=0.5)
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
    if bar_values:
        fig.update_yaxes(range=[0, max(bar_values) * 1.15], tickformat=",", row=1, col=1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=1, col=2)
    fig.update_xaxes(range=[0, MAX_SLOT], row=1, col=2)
    return fig

# =========================
# Figure: Penalty slots
# =========================
def build_penalty_figure(state: dict) -> go.Figure:
    epoch_str = str(state.get("epoch", 0))
    op_epoch  = state.get("operationalRecord", {}).get(epoch_str, {})
    mains = op_epoch.get("urlMainRecords", {})

    penalty_slots = {addr: meta.get("penaltySlots", 0) for addr, meta in mains.items()}
    items  = sorted(penalty_slots.items(), key=lambda kv: kv[1], reverse=True)
    addrs  = [addr for addr, _ in items]
    values = [v for _, v in items]
    names  = [LABEL.get(a, a) for a in addrs]
    total  = sum(values) if values else 1
    colors = [hex_to_rgba(PALETTE[i % len(PALETTE)], alpha=0.6) for i in range(len(values))]
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
    if values:
        fig.update_yaxes(range=[0, max(values) * 1.15], tickformat=",")
    return fig

# =========================
# Run
# =========================
if __name__ == "__main__":
    ensure_outdir(OUTPUT_DIR)

    # 0) Stake (PNG)
    stake_fig = build_stake_figure()
    stake_png = os.path.join(OUTPUT_DIR, f"stake_bars_epoch_{STAKE_EPOCH}_slot_{STAKE_SLOT}.png")
    stake_fig.write_image(stake_png, width=1200, height=600, scale=2)
    print(f"[OK] Saved: {stake_png}")

    # Carica stato per le altre figure
    state = load_state(FILENAME)
    epoch_str = str(state.get("epoch", 0))

    # 1) Summary combinato (PNG)
    combined = build_combined_figure(state)
    out_summary = os.path.join(OUTPUT_DIR, f"summary_epoch_{epoch_str}.png")
    combined.write_image(out_summary, width=1600, height=600, scale=2)
    print(f"[OK] Saved: {out_summary}")

    # 2) Penalty slots (PNG)
    penalty_fig = build_penalty_figure(state)
    out_penalty = os.path.join(OUTPUT_DIR, f"penalty_slots_epoch_{epoch_str}.png")
    penalty_fig.write_image(out_penalty, width=1000, height=600, scale=2)
    print(f"[OK] Saved: {out_penalty}")
