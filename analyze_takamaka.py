# analyze_takamaka.py
import os
import json
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors  # per HEX -> RGBA

# =========================
# Config
# =========================
FILENAME   = "three.state"
OUTPUT_DIR = "output"
MAX_SLOT   = 2399

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
# Plot 1: Stacked area 100% cumulata (senza bin)
# =========================
def plot_cumulative_share(addr_to_slots: dict, outpath: str):
    # Ordina indirizzi per etichetta leggibile
    addresses = sorted(addr_to_slots.keys(), key=lambda k: LABEL.get(k, k))

    T = MAX_SLOT + 1
    x = np.arange(T)

    # Matrice indicatori (n_nodes x T): 1 se il nodo valida quello slot
    indicators = []
    for addr in addresses:
        slots = [s for s in addr_to_slots.get(addr, []) if 0 <= s <= MAX_SLOT]
        v = np.zeros(T, dtype=int)
        if slots:
            v[slots] = 1
        indicators.append(v)
    indicators = np.vstack(indicators) if indicators else np.zeros((0, T), dtype=int)

    # Cumulate per nodo e totale
    cum_by_node = indicators.cumsum(axis=1)          # (n_nodes x T)
    cum_total   = cum_by_node.sum(axis=0)            # (T,)
    cum_total   = np.where(cum_total == 0, 1, cum_total)  # evita divisioni per zero

    # Quote cumulate (ogni colonna somma a 1)
    shares = cum_by_node / cum_total

    # Figura
    fig = go.Figure()
    for i, addr in enumerate(addresses):
        name = LABEL.get(addr, addr)
        line_color = PALETTE[i % len(PALETTE)]
        fill_color = hex_to_rgba(line_color, alpha=0.5)

        fig.add_trace(go.Scatter(
            x=x,
            y=shares[i],
            name=name,
            mode="lines",
            line=dict(color=line_color, width=2),
            stackgroup="one",
            fill="tonexty",
            fillcolor=fill_color,
            hovertemplate="Slot=%{x}<br>" + f"{name} share cumulata=%{{y:.1%}}<extra></extra>",
        ))

    fig.update_layout(
        title="Validated slots in percentage for each node",
        xaxis_title="Slot",
        yaxis_title="Validated slots in percentage",
        template="plotly_white",
        hovermode="x unified",
        legend_title_text="Nodo",
        margin=dict(l=60, r=40, t=70, b=50),
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    fig.update_xaxes(range=[0, MAX_SLOT])

    # Salva
    fig.write_image(outpath, width=1200, height=600, scale=2)
    print(f"[OK] Saved: {outpath}")

# =========================
# Plot 2: Vertical bar – delivered blocks per node
# =========================
def plot_delivered_blocks(op_rec_epoch: dict, epoch_str: str, outpath: str):
    overflows = op_rec_epoch.get("urlOverflowCounter", {})
    delivered = {addr: meta.get("deliveredBlocks", 0) for addr, meta in overflows.items()}
    if not delivered:
        print(f"[WARN] No deliveredBlocks for epoch {epoch_str}")
        return

    # Ordina e prepara
    items = sorted(delivered.items(), key=lambda kv: kv[1], reverse=True)
    addresses = [addr for addr, _ in items]
    values    = [v for _, v in items]
    names     = [LABEL.get(a, a) for a in addresses]
    total     = sum(values)
    perc_text = [f"{v:,.0f} ({v/total:.1%})" for v in values]
    colors    = [hex_to_rgba(PALETTE[i % len(PALETTE)], alpha=0.6) for i in range(len(values))]

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker=dict(color=colors, line=dict(width=1, color="black")),
        text=perc_text,
        textposition="outside",
        hovertemplate="%{x}<br>Blocks: %{y:,.0f}<br>Share: %{customdata:.1%}<extra></extra>",
        customdata=[v/total for v in values],
    ))

    fig.update_layout(
        title=f"Delivered blocks per node (epoch {epoch_str})",
        xaxis_title="Node",
        yaxis_title="Delivered blocks",
        template="plotly_white",
        margin=dict(l=80, r=80, t=80, b=80),
    )
    fig.update_yaxes(range=[0, max(values) * 1.15], tickformat=",")

    # Salva
    fig.write_image(outpath, width=1000, height=600, scale=2)
    print(f"[OK] Saved: {outpath}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    ensure_outdir(OUTPUT_DIR)
    state = load_state(FILENAME)
    epoch_str = str(state.get("epoch", 0))

    # --- Grafico 1: area 100% cumulata ---
    addr_to_slots = state.get("currentEpochSlotDistribution", {})
    out_area = os.path.join(OUTPUT_DIR, f"validated_slotsepoch_{epoch_str}.png")
    plot_cumulative_share(addr_to_slots, out_area)

    # --- Grafico 2: vertical bars deliveredBlocks ---
    op_epoch  = state.get("operationalRecord", {}).get(epoch_str, {})
    out_bars  = os.path.join(OUTPUT_DIR, f"delivered_blocks_epoch_{epoch_str}.png")
    plot_delivered_blocks(op_epoch, epoch_str, out_bars)
