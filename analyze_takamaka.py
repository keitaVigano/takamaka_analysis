import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------- Config & helpers -----------------------
NODES = [
    {"label":"node1",
     "main":"Be2ntZsJLQ0Kmp_KRv-281k-x5pWqnuUNNzFqsi35lQ.",
     "overflow":"_0OHvruLBmlRlDnCEmKdxyc65hBynAxRtiuQ8BHkrkM.",
     "stakeholder":"AD_VEGu6N6Vp4U4BfFddUn8FkoAbSWF61DFBQzl7jkA."},
    {"label":"node2",
     "main":"TgLaccJovUQPjgNcU1ZsS-tYrYnuNTRtjLdVvAqg3kI.",
     "overflow":"83MqfFUtbX6B1pgEYc71Sa7hfwoVmd3KDw7pawj3YDY.",
     "stakeholder":"lIpa4KfWG9NSAp4LmPrL62h7ysBpwj83Bw3FBheknVw."},
    {"label":"node3",
     "main":"Lb5sLN6alt5KcdMNE74onoOuQL8Y0eYgGZKqDOAUSTU.",
     "overflow":"y4V32jfDh8MyUviBvbzi_dMT54N_IqVxF8HtdWBfKiA.",
     "stakeholder":"ZjFWzoPAcM7LUbAxjI3PnoFyOkF78HSAOy2A19473t0."},
]
MAIN_LABELS = [f"main {n['label']}" for n in NODES]
OVER_LABELS  = [f"overflow {n['label']}" for n in NODES]

PALETTE = {"node1":"#1f77b4", "node2":"#2ca02c", "node3":"#d62728", "unknown":"#7f7f7f"}
COLOR_MAP = {
    **{f"main {n['label']}":     PALETTE[n["label"]] for n in NODES},
    **{f"overflow {n['label']}": PALETTE[n["label"]] for n in NODES},
    "unknown": PALETTE["unknown"],
}
COMP_PALETTE = {
    "Coinbase":"#636EFA", "Red fee":"#EF553B", "Green fee":"#00CC96",
    "Red frozen fee":"#AB63FA", "Green frozen fee":"#FFA15A"
}

def label_for_address(addr: str) -> str:
    if not isinstance(addr, str):
        return str(addr)
    for n in NODES:
        if addr == n["main"]:       return f"main {n['label']}"
        if addr == n["overflow"]:   return f"overflow {n['label']}"
        if addr == n["stakeholder"]:return f"stakeholder {n['label']}"
    return "unknown"

def load_state(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8", errors="ignore"))

def pick_prev_epoch(state: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    curr_epoch = state.get("epoch", None)
    op = state.get("operationalRecord", {}) or {}
    if isinstance(curr_epoch, int) and str(curr_epoch-1) in op:
        return curr_epoch-1, str(curr_epoch-1)
    keys = []
    for k in op.keys():
        try: keys.append(int(k))
        except: pass
    if not keys: return None, None
    keys.sort()
    if isinstance(curr_epoch, int):
        prior = [k for k in keys if k < curr_epoch]
        if prior: return prior[-1], str(prior[-1])
    return keys[-1], str(keys[-1])

def extract_prev_epoch_frames(state: Dict[str, Any], prev_epoch_key: str):
    op_prev = (state.get("operationalRecord", {}) or {}).get(prev_epoch_key, {}) or {}
    urlMainRecords     = op_prev.get("urlMainRecords", {}) or {}
    urlOverflowCounter = op_prev.get("urlOverflowCounter", {}) or {}

    rows_main = []
    for main_addr, v in urlMainRecords.items():
        rows_main.append({
            "main": main_addr,
            "label": label_for_address(main_addr),
            "nodeCoinbase": v.get("nodeCoinbase",0),
            "nodeRedFee": v.get("nodeRedFee",0),
            "nodeGreenFee": v.get("nodeGreenFee",0),
            "nodeRedFrozenFee": v.get("nodeRedFrozenFee",0),
            "nodeGreenFrozenFee": v.get("nodeGreenFrozenFee",0),
            "penaltySlots": v.get("penaltySlots",0),
            "nodePublicKey": v.get("nodePublicKey"),
            "nodeHexSignerHash": v.get("nodeHexSignerHash"),
        })
    df_main_prev = pd.DataFrame(rows_main)

    rows_over = []
    for over_addr, v in urlOverflowCounter.items():
        rows_over.append({
            "overflow": over_addr,
            "label": label_for_address(over_addr),
            "deliveredBlocks": v.get("deliveredBlocks",0),
            "missedBlocks": v.get("missedBlocks",0),
            "mainUrl64Addr": v.get("mainUrl64Addr"),
        })
    df_over_prev = pd.DataFrame(rows_over)

    dist_curr = state.get("currentEpochSlotDistribution", {}) or {}
    rows_curr = [{"main":m,"label":label_for_address(m),"slots_list":(slots if isinstance(slots,list) else [])}
                 for m,slots in dist_curr.items()]
    df_stake_curr = pd.DataFrame(rows_curr)
    return df_main_prev, df_over_prev, df_stake_curr

def estimate_prev_slot_assignment(state: Dict[str, Any], prev_epoch_int: Optional[int]) -> pd.DataFrame:
    pk = state.get("proposedKeys", {}) or {}
    rows = []
    if prev_epoch_int is None:
        return pd.DataFrame()
    for _, v in pk.items():
        try:
            if int(v.get("epoch",-1)) != prev_epoch_int: continue
        except: continue
        rows.append({
            "nodeKey": v.get("nodePublicKey") or v.get("nodeHexSignerHash") or "unknown",
            "slot": v.get("slot")
        })
    return pd.DataFrame(rows)

def build_cumulative_long(df_slots: pd.DataFrame, key_to_main: Dict[str,str], label_order: List[str]) -> pd.DataFrame:
    if df_slots.empty: return pd.DataFrame(columns=["slot","label","cumulative"])
    df = df_slots.copy()
    df["main"]  = df["nodeKey"].map(key_to_main).fillna("unknown")
    df["label"] = df["main"].apply(label_for_address)
    df = df[pd.to_numeric(df["slot"], errors="coerce").notna()].copy()
    df["slot"] = df["slot"].astype(int)
    max_slot = int(df["slot"].max())
    slots = pd.DataFrame({"slot": np.arange(0, max_slot+1, 1, dtype=int)})
    counts = df.groupby(["slot","label"], as_index=False).size().rename(columns={"size":"count"})
    wide = counts.pivot(index="slot", columns="label", values="count").reindex(slots["slot"]).fillna(0)
    cols = [c for c in label_order if c in wide.columns] + [c for c in wide.columns if c not in label_order]
    wide = wide[cols]
    cum = wide.cumsum().reset_index().melt(id_vars="slot", var_name="label", value_name="cumulative")
    return cum

# ---------- data builder (riusabile da export e dashboard) ----------
def build_data(state_path: Path):
    state = load_state(state_path)
    prev_epoch_int, prev_epoch_key = pick_prev_epoch(state)
    df_main_prev, df_over_prev, df_stake_curr = extract_prev_epoch_frames(state, prev_epoch_key)

    # ensure known nodes present (zeros allowed)
    all_mains = pd.DataFrame([{"main":n["main"], "label":f"main {n['label']}"} for n in NODES])
    dm = all_mains.merge(df_main_prev, on=["main","label"], how="left").fillna(0)
    dm["nodeRewards_total"] = (dm["nodeCoinbase"] + dm["nodeRedFee"] + dm["nodeGreenFee"]
                               + dm["nodeRedFrozenFee"] + dm["nodeGreenFrozenFee"])
    all_overflows = pd.DataFrame([{"overflow":n["overflow"], "label":f"overflow {n['label']}"} for n in NODES])
    dof = all_overflows.merge(
        df_over_prev[["overflow","label","deliveredBlocks"]] if not df_over_prev.empty else
        pd.DataFrame(columns=["overflow","label","deliveredBlocks"]),
        on=["overflow","label"], how="left"
    ).fillna({"deliveredBlocks":0})
    total_delivered = dof["deliveredBlocks"].sum()
    dof["pct_delivered"] = (dof["deliveredBlocks"]/total_delivered*100) if total_delivered>0 else 0.0

    # cumulative prev
    key_to_main = {}
    for _, r in df_main_prev.iterrows():
        if pd.notna(r.get("nodePublicKey")):    key_to_main[r["nodePublicKey"]]    = r["main"]
        if pd.notna(r.get("nodeHexSignerHash")):key_to_main[r["nodeHexSignerHash"]]= r["main"]
    df_prev_slots = estimate_prev_slot_assignment(state, prev_epoch_int)
    cum_prev = build_cumulative_long(df_prev_slots, key_to_main, MAIN_LABELS+["unknown"])

    # cumulative current
    curr_rows = []
    if not df_stake_curr.empty:
        for _, r in df_stake_curr.iterrows():
            for s in (r["slots_list"] or []):
                curr_rows.append({"nodeKey": r["main"], "slot": s})
    df_curr_slots = pd.DataFrame(curr_rows)
    cum_curr = build_cumulative_long(df_curr_slots, {}, MAIN_LABELS)

    # rewards stacked long
    comp_long = pd.DataFrame({
        "label": np.repeat(dm["label"].astype(str).values, 5),
        "component": (["Coinbase"]*len(dm) + ["Red fee"]*len(dm) + ["Green fee"]*len(dm)
                      + ["Red frozen fee"]*len(dm) + ["Green frozen fee"]*len(dm)),
        "value": np.concatenate([
            dm["nodeCoinbase"].values,
            dm["nodeRedFee"].values,
            dm["nodeGreenFee"].values,
            dm["nodeRedFrozenFee"].values,
            dm["nodeGreenFrozenFee"].values
        ], axis=0)
    })

    return dict(
        state=state, prev_epoch_int=prev_epoch_int,
        dm=dm, dof=dof, cum_prev=cum_prev, cum_curr=cum_curr, comp_long=comp_long
    )

# ---------- figures builder ----------
def make_figures(data: Dict[str, Any]):
    prev_epoch_int = data["prev_epoch_int"]
    dm, dof, cum_prev, cum_curr, comp_long = data["dm"], data["dof"], data["cum_prev"], data["cum_curr"], data["comp_long"]

    # delivered percent
    fig_delivered_pct = go.Figure()
    for lbl in OVER_LABELS:
        sub = dof[dof["label"].astype(str) == lbl]
        fig_delivered_pct.add_bar(
            x=sub["label"], y=sub["pct_delivered"], name=str(lbl),
            text=sub["pct_delivered"].map(lambda x: f"{x:.2f}%"),
            marker_color=COLOR_MAP.get(str(lbl), PALETTE["unknown"]),
            hovertemplate="%{x}<br>%{y:.2f}% (%{customdata:,} blocks)<extra></extra>",
            customdata=np.array(sub["deliveredBlocks"])
        )
    fig_delivered_pct.update_layout(
        title=f"Delivered blocks per overflow — epoch {prev_epoch_int} (percent)",
        xaxis_title="", yaxis_title="Delivered blocks (%)",
        showlegend=False, paper_bgcolor="white", plot_bgcolor="white"
    )

    # delivered absolute
    fig_delivered_abs = go.Figure()
    for lbl in OVER_LABELS:
        sub = dof[dof["label"].astype(str) == lbl]
        fig_delivered_abs.add_bar(
            x=sub["label"], y=sub["deliveredBlocks"], name=str(lbl),
            text=sub["deliveredBlocks"], marker_color=COLOR_MAP.get(str(lbl), PALETTE["unknown"]),
            hovertemplate="%{x}<br>Delivered: %{y:,} blocks<extra></extra>"
        )
    fig_delivered_abs.update_layout(
        title=f"Delivered blocks per overflow — epoch {prev_epoch_int} (absolute)",
        xaxis_title="", yaxis_title="Delivered blocks (count)",
        showlegend=False, paper_bgcolor="white", plot_bgcolor="white"
    )

    # slots prev vs curr
    fig_slots = make_subplots(rows=1, cols=2,
                              subplot_titles=("Previous epoch (cumulative)", "Current epoch (cumulative)"))
    if not cum_prev.empty:
        for lab in cum_prev["label"].unique():
            sub = cum_prev[cum_prev["label"]==lab]
            fig_slots.add_trace(
                go.Scatter(x=sub["slot"], y=sub["cumulative"], name=lab, stackgroup="prev",
                           line=dict(width=0.5), marker=dict(color=COLOR_MAP.get(lab, PALETTE["unknown"])),
                           hovertemplate=lab + "<br>Slot %{x}: %{y:,} cumulative<extra></extra>"),
                row=1, col=1
            )
    else:
        fig_slots.add_annotation(text="No data", xref="x1", yref="y1", x=0.5, y=0.5, showarrow=False)

    if not cum_curr.empty:
        for lab in cum_curr["label"].unique():
            sub = cum_curr[cum_curr["label"]==lab]
            fig_slots.add_trace(
                go.Scatter(x=sub["slot"], y=sub["cumulative"], name=lab, stackgroup="curr",
                           line=dict(width=0.5), marker=dict(color=COLOR_MAP.get(lab, PALETTE["unknown"])),
                           hovertemplate=lab + "<br>Slot %{x}: %{y:,} cumulative<extra></extra>"),
                row=1, col=2
            )
    else:
        fig_slots.add_annotation(text="No data", xref="x2", yref="y2", x=0.5, y=0.5, showarrow=False)
    fig_slots.update_layout(title_text="Cumulative slot assignments — PREVIOUS vs CURRENT",
                            paper_bgcolor="white", plot_bgcolor="white", showlegend=True)

    # rewards total
    fig_rewards = px.bar(
        dm, x="label", y="nodeRewards_total", color="label",
        color_discrete_map=COLOR_MAP,
        title=f"Node rewards per main (coinbase + fees) — epoch {prev_epoch_int}",
        labels={"label":"", "nodeRewards_total":"Amount (token units)"}
    )
    custom = np.stack([
        dm["nodeCoinbase"], dm["nodeRedFee"], dm["nodeGreenFee"],
        dm["nodeRedFrozenFee"], dm["nodeGreenFrozenFee"], dm["nodeRewards_total"]
    ], axis=-1)
    fig_rewards.update_traces(
        hovertemplate=(
            "%{x}<br>"
            "Coinbase: %{customdata[0]:,}<br>"
            "Red fee: %{customdata[1]:,}<br>"
            "Green fee: %{customdata[2]:,}<br>"
            "Red frozen fee: %{customdata[3]:,}<br>"
            "Green frozen fee: %{customdata[4]:,}<br>"
            "<b>Total: %{customdata[5]:,}</b><extra></extra>"
        ),
        customdata=custom
    )
    fig_rewards.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")

    # rewards stacked
    comp_long = data["comp_long"]
    fig_rewards_stacked = px.bar(
        comp_long, x="label", y="value", color="component", barmode="stack",
        color_discrete_map=COMP_PALETTE,
        title=f"Rewards breakdown per main — epoch {prev_epoch_int}",
        labels={"label":"", "value":"Amount (token units)", "component":""}
    )
    fig_rewards_stacked.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:,}<extra></extra>")
    fig_rewards_stacked.update_layout(paper_bgcolor="white", plot_bgcolor="white")

    # penalty slots
    fig_penalty = px.bar(
        dm, x="label", y="penaltySlots", text="penaltySlots",
        color="label", color_discrete_map=COLOR_MAP,
        title=f"Penalty slots per main — epoch {prev_epoch_int}",
        labels={"label":"", "penaltySlots":"Penalty slots"}
    )
    fig_penalty.update_traces(textposition="outside", hovertemplate="%{x}<br>Penalty slots: %{y:,}<extra></extra>")
    max_y = (dm["penaltySlots"].max() if len(dm) else 0)
    fig_penalty.update_yaxes(range=[0, max_y*1.2 if max_y else 1])
    fig_penalty.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")

    return {
        "delivered_pct": fig_delivered_pct,
        "delivered_abs": fig_delivered_abs,
        "slots": fig_slots,
        "rewards_total": fig_rewards,
        "rewards_stacked": fig_rewards_stacked,
        "penalty": fig_penalty
    }

# ---------- PNG export ----------
def export_pngs(figs: Dict[str, go.Figure], outdir: Path, prev_epoch_int: int,
                width=1600, height=900, scale=2) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    names = {
        "delivered_pct":   f"delivered_overflow_percent_epoch{prev_epoch_int}.png",
        "delivered_abs":   f"delivered_overflow_absolute_epoch{prev_epoch_int}.png",
        "slots":           f"cumulative_slots_prev_vs_curr_epoch{prev_epoch_int}.png",
        "rewards_total":   f"rewards_total_per_main_epoch{prev_epoch_int}.png",
        "rewards_stacked": f"rewards_breakdown_per_main_epoch{prev_epoch_int}.png",
        "penalty":         f"penalty_slots_per_main_epoch{prev_epoch_int}.png",
    }
    saved = []
    for key, fig in figs.items():
        p = outdir / names[key]
        fig.write_image(str(p), width=width, height=height, scale=scale)  # PNG only
        saved.append(p)
    return saved

# =========================== CLI / main ===========================
parser = argparse.ArgumentParser(description="Takamaka dashboard (Dash) + PNG export")
parser.add_argument("--state", default="one.state", help="Percorso al file .state")
parser.add_argument("--serve", action="store_true", help="Avvia la dashboard interattiva")
parser.add_argument("--export-only", action="store_true", help="Solo export PNG e termina")
parser.add_argument("--timeframe", default=None, help="Sottocartella in output/")
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
parser.add_argument("--scale", type=int, default=2)
args = parser.parse_args()

data = build_data(Path(args.state))
prev_epoch_int = data["prev_epoch_int"]
figs = make_figures(data)

# cartella timeframe
default_timeframe = f"epoch{prev_epoch_int}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
timeframe = args.timeframe or default_timeframe
outdir = Path("output") / timeframe

if args.export-only and not args.serve:
    exported = export_pngs(figs, outdir, prev_epoch_int, args.width, args.height, args.scale)
    print("Saved PNGs:")
    for p in exported: print("-", p.resolve())
else:
    # ------- serve dashboard (con pulsante Export PNGs) -------
    app = Dash(__name__)
    app.title = f"Takamaka Dashboard — prev epoch {prev_epoch_int}"

    app.layout = html.Div([
        html.H2(f"Takamaka — Previous epoch {prev_epoch_int}"),
        html.Div([
            html.Button("Export PNGs", id="btn-export", n_clicks=0, style={"marginRight":"12px"}),
            html.Span(id="export-msg", style={"marginRight":"24px"}),
            dcc.Dropdown(
                id="node-filter",
                options=[{"label":"All","value":"all"}] + [{"label":n["label"], "value":f"main {n['label']}"} for n in NODES],
                value="all", clearable=False, style={"width":"240px","display":"inline-block","marginRight":"12px"}
            ),
            dcc.RadioItems(
                id="deliv-mode",
                options=[{"label":"Percent","value":"pct"}, {"label":"Absolute","value":"abs"}],
                value="pct", inline=True
            ),
        ], style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"12px"}),

        dcc.Tabs([
            dcc.Tab(label="Delivered (overflow)", children=[dcc.Graph(id="fig-delivered")]),
            dcc.Tab(label="Slots cumulative (prev vs curr)", children=[dcc.Graph(id="fig-slots")]),
            dcc.Tab(label="Rewards total (main)", children=[dcc.Graph(id="fig-rewards")]),
            dcc.Tab(label="Rewards breakdown (stacked)", children=[dcc.Graph(id="fig-rewards-stacked")]),
            dcc.Tab(label="Penalty slots (main)", children=[dcc.Graph(id="fig-penalty")]),
        ]),
        # store parametri export
        dcc.Store(id="store-outdir", data=str(outdir)),
        dcc.Store(id="store-width", data=args.width),
        dcc.Store(id="store-height", data=args.height),
        dcc.Store(id="store-scale", data=args.scale),
    ], style={"padding":"12px"})

    # --- callbacks grafici (riuso dati/figure già creati) ---
    # Delivered
    @app.callback(
        Output("fig-delivered","figure"),
        Input("node-filter","value"),
        Input("deliv-mode","value"),
    )
    def update_delivered(node_value, mode):
        df = data["dof"].copy()
        if node_value != "all":
            node = node_value.replace("main ","overflow ")
            df = df[df["label"] == node]
        if mode == "pct":
            fig = px.bar(df, x="label", y="pct_delivered", text=df["pct_delivered"].map(lambda x: f"{x:.2f}%"),
                         color="label", color_discrete_map=COLOR_MAP,
                         labels={"label":"","pct_delivered":"Delivered blocks (%)"},
                         title="Delivered blocks per overflow (percent)")
            fig.update_traces(textposition="outside",
                              hovertemplate="%{x}<br>%{y:.2f}% (%{customdata:,} blocks)<extra></extra>",
                              customdata=np.array(df["deliveredBlocks"]))
        else:
            fig = px.bar(df, x="label", y="deliveredBlocks",
                         color="label", color_discrete_map=COLOR_MAP,
                         labels={"label":"","deliveredBlocks":"Delivered blocks (count)"},
                         title="Delivered blocks per overflow (absolute)")
            fig.update_traces(hovertemplate="%{x}<br>Delivered: %{y:,} blocks<extra></extra>")
        fig.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white"))
        return fig

    # Slots
    @app.callback(
        Output("fig-slots","figure"),
        Input("node-filter","value"),
    )
    def update_slots(node_value):
        cum_prev, cum_curr = data["cum_prev"], data["cum_curr"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Previous epoch (cumulative)","Current epoch (cumulative)"))
        def add_stack(df_long, col_idx):
            if df_long.empty:
                fig.add_annotation(text="No data", xref=f"x{col_idx}", yref=f"y{col_idx}", x=0.5, y=0.5, showarrow=False)
                return
            labels = df_long["label"].unique()
            for lab in labels:
                sub = df_long[df_long["label"]==lab]
                vis = True if node_value=="all" else (lab == node_value)
                fig.add_trace(
                    go.Scatter(x=sub["slot"], y=sub["cumulative"], name=lab, stackgroup=f"sg{col_idx}",
                               line=dict(width=0.5), marker=dict(color=COLOR_MAP.get(lab, PALETTE["unknown"])),
                               hovertemplate=lab + "<br>Slot %{x}: %{y:,} cumulative<extra></extra>",
                               visible=vis),
                    row=1, col=col_idx
                )
        add_stack(cum_prev, 1)
        add_stack(cum_curr, 2)
        fig.update_layout(title_text="Cumulative slot assignments — PREVIOUS vs CURRENT",
                          paper_bgcolor="white", plot_bgcolor="white", showlegend=True)
        return fig

    # Rewards total
    @app.callback(
        Output("fig-rewards","figure"),
        Input("node-filter","value"),
    )
    def update_rewards(node_value):
        df = data["dm"].copy()
        if node_value != "all":
            df = df[df["label"] == node_value]
        fig = px.bar(df, x="label", y="nodeRewards_total", color="label",
                     color_discrete_map=COLOR_MAP,
                     labels={"label":"", "nodeRewards_total":"Amount (token units)"},
                     title="Node rewards per main (coinbase + fees)")
        custom = np.stack([df["nodeCoinbase"], df["nodeRedFee"], df["nodeGreenFee"],
                           df["nodeRedFrozenFee"], df["nodeGreenFrozenFee"], df["nodeRewards_total"]], axis=-1)
        fig.update_traces(
            hovertemplate=(
                "%{x}<br>"
                "Coinbase: %{customdata[0]:,}<br>"
                "Red fee: %{customdata[1]:,}<br>"
                "Green fee: %{customdata[2]:,}<br>"
                "Red frozen fee: %{customdata[3]:,}<br>"
                "Green frozen fee: %{customdata[4]:,}<br>"
                "<b>Total: %{customdata[5]:,}</b><extra></extra>"
            ),
            customdata=custom
        )
        fig.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
        return fig

    # Rewards stacked
    @app.callback(
        Output("fig-rewards-stacked","figure"),
        Input("node-filter","value"),
    )
    def update_rewards_stacked(node_value):
        df = data["comp_long"].copy()
        if node_value != "all":
            df = df[df["label"] == node_value]
        fig = px.bar(df, x="label", y="value", color="component", barmode="stack",
                     color_discrete_map=COMP_PALETTE,
                     labels={"label":"", "value":"Amount (token units)", "component":""},
                     title="Rewards breakdown per main (stacked)")
        fig.update_traces(hovertemplate="%{x}<br>%{legendgroup}: %{y:,}<extra></extra>")
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        return fig

    # Penalty
    @app.callback(
        Output("fig-penalty","figure"),
        Input("node-filter","value"),
    )
    def update_penalty(node_value):
        df = data["dm"].copy()
        if node_value != "all":
            df = df[df["label"] == node_value]
        fig = px.bar(df, x="label", y="penaltySlots", text="penaltySlots",
                     color="label", color_discrete_map=COLOR_MAP,
                     labels={"label":"", "penaltySlots":"Penalty slots"},
                     title="Penalty slots per main")
        fig.update_traces(textposition="outside", hovertemplate="%{x}<br>Penalty slots: %{y:,}<extra></extra>")
        max_y = (df["penaltySlots"].max() if len(df) else 0)
        fig.update_yaxes(range=[0, max_y*1.2 if max_y else 1])
        fig.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
        return fig

    # ---- Export button callback ----
    @app.callback(
        Output("export-msg","children"),
        Input("btn-export","n_clicks"),
        State("store-outdir","data"),
        State("store-width","data"),
        State("store-height","data"),
        State("store-scale","data"),
        prevent_initial_call=True
    )
    def on_export(n_clicks, outdir_str, w, h, s):
        if not n_clicks:
            return no_update
        saved = export_pngs(figs, Path(outdir_str), prev_epoch_int, w, h, s)
        return f"Saved {len(saved)} PNGs → {Path(outdir_str).resolve()}"

    app.run_server(debug=False)
