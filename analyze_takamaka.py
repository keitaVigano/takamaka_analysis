# analyze_takamaka_prev_epoch.py
# Genera CSV e grafici per l'epoch precedente rispetto allo snapshot del file di stato.
# Tutti gli output in: output/<timeframe>/...
#
# Migliorie grafiche:
# - figure più grandi, griglia Y
# - valori numerici sopra le barre
# - formattazione Y con separatore migliaia
# - padding/margini per evitare tagli
#
# Requisiti: python>=3.8, pandas, matplotlib
#   pip install pandas matplotlib

import json
import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


# ======= CONFIG: indirizzi noti (etichette leggibili per i grafici) =======
NODES = [
    {
        "label": "node1",
        "main": "Be2ntZsJLQ0Kmp_KRv-281k-x5pWqnuUNNzFqsi35lQ.",
        "overflow": "_0OHvruLBmlRlDnCEmKdxyc65hBynAxRtiuQ8BHkrkM.",
        "stakeholder": "AD_VEGu6N6Vp4U4BfFddUn8FkoAbSWF61DFBQzl7jkA."
    },
    {
        "label": "node2",
        "main": "TgLaccJovUQPjgNcU1ZsS-tYrYnuNTRtjLdVvAqg3kI.",
        "overflow": "83MqfFUtbX6B1pgEYc71Sa7hfwoVmd3KDw7pawj3YDY.",
        "stakeholder": "lIpa4KfWG9NSAp4LmPrL62h7ysBpwj83Bw3FBheknVw."
    },
    {
        "label": "node3",
        "main": "Lb5sLN6alt5KcdMNE74onoOuQL8Y0eYgGZKqDOAUSTU.",
        "overflow": "y4V32jfDh8MyUviBvbzi_dMT54N_IqVxF8HtdWBfKiA.",
        "stakeholder": "ZjFWzoPAcM7LUbAxjI3PnoFyOkF78HSAOy2A19473t0."
    },
]
# ==========================================================================


def label_for_address(addr: str) -> str:
    if not isinstance(addr, str):
        return str(addr)
    for n in NODES:
        if addr == n["main"]:
            return f"main {n['label']}"
        if addr == n["overflow"]:
            return f"overflow {n['label']}"
        if addr == n["stakeholder"]:
            return f"stakeholder {n['label']}"
    return addr


def load_state(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def pick_prev_epoch(state: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    """Determina l'epoch 'corrente' del dump e sceglie la chiave dell'epoch precedente in operationalRecord."""
    curr_epoch = state.get("epoch", None)
    op = state.get("operationalRecord", {}) or {}

    if isinstance(curr_epoch, int):
        prev_epoch = curr_epoch - 1
        if str(prev_epoch) in op:
            return prev_epoch, str(prev_epoch)

    keys_int = []
    for k in op.keys():
        try:
            keys_int.append(int(k))
        except Exception:
            pass
    if not keys_int:
        return None, None

    keys_int.sort()
    if isinstance(curr_epoch, int):
        candidates = [k for k in keys_int if k < curr_epoch]
        if candidates:
            return candidates[-1], str(candidates[-1])

    return keys_int[-1], str(keys_int[-1])


def extract_prev_epoch_frames(state: Dict[str, Any], prev_epoch_key: str):
    """Estrae i DataFrame per l'epoch precedente (prev_epoch_key)."""
    op = state.get("operationalRecord", {}) or {}
    op_prev = op.get(prev_epoch_key, {}) or {}

    urlMainRecords = op_prev.get("urlMainRecords", {}) or {}
    urlOverflowCounter = op_prev.get("urlOverflowCounter", {}) or {}

    # MAIN (epoch prev)
    rows_main = []
    for main_addr, vals in urlMainRecords.items():
        rows_main.append({
            "main": main_addr,
            "holderRedFee": vals.get("holderRedFee", 0),
            "holderGreenFee": vals.get("holderGreenFee", 0),
            "nodeRedFee": vals.get("nodeRedFee", 0),
            "nodeGreenFee": vals.get("nodeGreenFee", 0),
            "holderCoinbase": vals.get("holderCoinbase", 0),
            "nodeCoinbase": vals.get("nodeCoinbase", 0),
            "holderRedFrozenFee": vals.get("holderRedFrozenFee", 0),
            "holderGreenFrozenFee": vals.get("holderGreenFrozenFee", 0),
            "nodeRedFrozenFee": vals.get("nodeRedFrozenFee", 0),
            "nodeGreenFrozenFee": vals.get("nodeGreenFrozenFee", 0),
            "penaltySlots": vals.get("penaltySlots", 0),
            "nodePublicKey": vals.get("nodePublicKey"),
            "nodeHexSignerHash": vals.get("nodeHexSignerHash"),
        })
    df_main_prev = pd.DataFrame(rows_main)
    if not df_main_prev.empty:
        df_main_prev["label"] = df_main_prev["main"].apply(label_for_address)

    # OVERFLOW (epoch prev)
    rows_over = []
    for over_addr, vals in urlOverflowCounter.items():
        rows_over.append({
            "overflow": over_addr,
            "deliveredBlocks": vals.get("deliveredBlocks", 0),
            "missedBlocks": vals.get("missedBlocks", 0),
            "mainUrl64Addr": vals.get("mainUrl64Addr"),
            "lowUrl64Addr": vals.get("lowUrl64Addr"),
        })
    df_over_prev = pd.DataFrame(rows_over)
    if not df_over_prev.empty:
        df_over_prev["label"] = df_over_prev["overflow"].apply(label_for_address)

    # STAKE (epoch corrente) come riferimento
    dist_curr = state.get("currentEpochSlotDistribution", {}) or {}
    rows_stake_curr = []
    for main_addr, slots in dist_curr.items():
        slots_count = len(slots) if isinstance(slots, list) else 0
        rows_stake_curr.append({"main": main_addr, "stakeSlots_epochCurrent": slots_count})
    df_stake_curr = pd.DataFrame(rows_stake_curr)
    if not df_stake_curr.empty:
        df_stake_curr["label"] = df_stake_curr["main"].apply(label_for_address)

    return df_main_prev, df_over_prev, df_stake_curr


def estimate_prev_slot_assignment(state: Dict[str, Any], prev_epoch_int: Optional[int]) -> pd.DataFrame:
    """Stima gli slot assegnati nell'epoch precedente usando proposedKeys (se presenti)."""
    pk = state.get("proposedKeys", {}) or {}
    rows = []
    if prev_epoch_int is None:
        return pd.DataFrame(rows)

    for _, v in pk.items():
        try:
            if int(v.get("epoch", -9999)) != prev_epoch_int:
                continue
        except Exception:
            continue
        node_pub = v.get("nodePublicKey") or v.get("nodeHexSignerHash") or "unknown"
        rows.append({
            "nodeKey": node_pub,
            "slot": v.get("slot"),
            "burned": v.get("burned", False),
        })
    if not rows:
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)
    return df.groupby("nodeKey", as_index=False).agg(assignedSlots_est=("slot", "count"))


def group_overflow_by_main(df_over_prev: pd.DataFrame) -> pd.DataFrame:
    """Somma delivered/missed per mainUrl64Addr (epoch prev)."""
    if df_over_prev.empty:
        return df_over_prev
    grp = (
        df_over_prev
        .groupby("mainUrl64Addr", as_index=False)[["deliveredBlocks", "missedBlocks"]]
        .sum()
        .rename(columns={"mainUrl64Addr": "main"})
    )
    grp["label"] = grp["main"].apply(label_for_address)
    return grp


def filter_known_nodes(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mantiene solo righe con indirizzi presenti in NODES (etichette pulite)."""
    if df.empty:
        return df
    known = {n["main"] for n in NODES} | {n["overflow"] for n in NODES}
    return df[df[column].isin(known)].copy()


# ---------- Helpers per grafici leggibili ----------
def _fmt_thousands(x, pos=None):
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def _add_value_labels(ax, bars, use_human=False):
    """Scrive il valore sopra ogni barra."""
    for b in bars:
        h = b.get_height()
        if h is None or h == 0:
            continue
        if use_human and abs(h) >= 1_000_000_000:
            # etichetta compatta per numeri enormi
            val = _human(h)
        else:
            val = f"{int(h):,}"
        ax.annotate(
            val,
            xy=(b.get_x() + b.get_width()/2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=9
        )

def _human(v: float) -> str:
    """Formato compatto (K/M/B/T) solo per etichette se molto grandi."""
    units = ["", "K", "M", "B", "T"]
    i = 0
    v = float(v)
    while abs(v) >= 1000 and i < len(units) - 1:
        v /= 1000.0
        i += 1
    return f"{v:.2f}{units[i]}"

def plot_bar(x, y, title, xlabel, ylabel, outpath: Path, humanize_labels=False):
    """Un grafico per figura, senza specificare colori, con etichette valori e Y formattata."""
    fig = plt.figure(figsize=(9, 5), dpi=120)
    ax = fig.gca()
    bars = ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_thousands))
    plt.xticks(rotation=25, ha="right")
    # padding in alto per non tagliare le etichette
    ymax = max(y) if len(y) else 0
    ax.set_ylim(0, ymax * 1.12 if ymax else 1)
    _add_value_labels(ax, bars, use_human=humanize_labels)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Takamaka (previous epoch) and save under output/<timeframe>/")
    parser.add_argument("--state", default="one.state", help="Percorso al file di stato (default: one.state)")
    parser.add_argument("--timeframe", default=None, help="Sottocartella dentro output/ (default: epoch<prev>)")
    args = parser.parse_args()

    state_path = Path(args.state)
    if not state_path.exists():
        print(f"ERRORE: file di stato non trovato: {state_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    state = load_state(state_path)
    prev_epoch_int, prev_epoch_key = pick_prev_epoch(state)
    if prev_epoch_key is None:
        print("ERRORE: impossibile individuare l'epoch precedente nei record operativi.", file=sys.stderr)
        sys.exit(2)

    timeframe = args.timeframe or f"epoch{prev_epoch_int}"
    outdir = Path("output") / timeframe
    outdir.mkdir(parents=True, exist_ok=True)

    # Estrazione tabelle epoch precedente
    df_main_prev, df_over_prev, df_stake_curr = extract_prev_epoch_frames(state, prev_epoch_key)

    # Aggregazione overflow per MAIN
    df_over_by_main_prev = group_overflow_by_main(df_over_prev)

    # Filtri (mostra solo i 3 nodi noti, se presenti)
    df_main_prev_known = filter_known_nodes(df_main_prev, "main")
    df_over_prev_known = filter_known_nodes(df_over_prev, "overflow")
    df_over_by_main_prev_known = filter_known_nodes(df_over_by_main_prev, "main")
    df_stake_curr_known = filter_known_nodes(df_stake_curr, "main")

    # Stima slot assegnati nell'epoch precedente (proposedKeys)
    df_prev_slot_est = estimate_prev_slot_assignment(state, prev_epoch_int)

    # === Salvataggi CSV ===
    p_main_csv = outdir / f"epoch{prev_epoch_int}_main_rewards_penalties.csv"
    p_over_csv = outdir / f"epoch{prev_epoch_int}_overflow_counters.csv"
    p_over_by_main_csv = outdir / f"epoch{prev_epoch_int}_overflow_by_main.csv"
    p_stake_curr_csv = outdir / "epochCurrent_stakeSlots_reference.csv"
    p_prev_slot_est_csv = outdir / f"epoch{prev_epoch_int}_slot_assignment_estimate.csv"

    (df_main_prev_known if not df_main_prev_known.empty else df_main_prev).to_csv(p_main_csv, index=False)
    (df_over_prev_known if not df_over_prev_known.empty else df_over_prev).to_csv(p_over_csv, index=False)
    (df_over_by_main_prev_known if not df_over_by_main_prev_known.empty else df_over_by_main_prev).to_csv(p_over_by_main_csv, index=False)
    (df_stake_curr_known if not df_stake_curr_known.empty else df_stake_curr).to_csv(p_stake_curr_csv, index=False)
    df_prev_slot_est.to_csv(p_prev_slot_est_csv, index=False)

    # === Grafici (epoch precedente) ===
    dm = (df_main_prev_known if not df_main_prev_known.empty else df_main_prev)
    if not dm.empty:
        plot_bar(
            dm["label"] if "label" in dm.columns else dm["main"],
            dm["penaltySlots"],
            f"Penalty slots per main (epoch {prev_epoch_int})",
            "Main nodes",
            "Penalty slots",
            outdir / f"epoch{prev_epoch_int}_penalty_slots_per_main.png",
            humanize_labels=False,
        )

        dm2 = dm.copy()
        dm2["nodeRewards_total"] = (
            dm2.get("nodeCoinbase", 0)
            + dm2.get("nodeRedFee", 0)
            + dm2.get("nodeGreenFee", 0)
            + dm2.get("nodeRedFrozenFee", 0)
            + dm2.get("nodeGreenFrozenFee", 0)
        )
        plot_bar(
            dm2["label"] if "label" in dm2.columns else dm2["main"],
            dm2["nodeRewards_total"],
            f"Node rewards (coinbase+fees) per main (epoch {prev_epoch_int})",
            "Main nodes",
            "Amount (token units)",
            outdir / f"epoch{prev_epoch_int}_node_rewards_per_main.png",
            humanize_labels=True,  # per numeri molto grandi etichetta compatta
        )

    dof = (df_over_prev_known if not df_over_prev_known.empty else df_over_prev)
    if not dof.empty:
        plot_bar(
            dof["label"] if "label" in dof.columns else dof["overflow"],
            dof["deliveredBlocks"],
            f"Delivered blocks per overflow (epoch {prev_epoch_int})",
            "Overflow nodes",
            "Delivered blocks",
            outdir / f"epoch{prev_epoch_int}_delivered_blocks_per_overflow.png",
            humanize_labels=False,
        )

    # === Grafico (epoch corrente - riferimento) ===
    ds = (df_stake_curr_known if not df_stake_curr_known.empty else df_stake_curr)
    if not ds.empty:
        plot_bar(
            ds["label"] if "label" in ds.columns else ds["main"],
            ds["stakeSlots_epochCurrent"],
            "Stake slots per main (epoch corrente - riferimento)",
            "Main nodes",
            "Assigned slots",
            outdir / "epochCurrent_stake_slots_reference.png",
            humanize_labels=False,
        )

    # Log percorsi creati
    print("\n== FILE SALVATI ==")
    for p in [p_main_csv, p_over_csv, p_over_by_main_csv, p_stake_curr_csv, p_prev_slot_est_csv]:
        print("-", p.resolve())
    for img in [
        outdir / f"epoch{prev_epoch_int}_penalty_slots_per_main.png",
        outdir / f"epoch{prev_epoch_int}_node_rewards_per_main.png",
        outdir / f"epoch{prev_epoch_int}_delivered_blocks_per_overflow.png",
        outdir / "epochCurrent_stake_slots_reference.png",
    ]:
        if img.exists():
            print("-", img.resolve())

    print("\nNote:")
    print("- Ricompense/penalità dell’epoch precedente: operationalRecord[prev].")
    print("- Delivered/missed per overflow: urlOverflowCounter (epoch prev) + aggregazione per main.")
    print("- Lo 'stake' dell’epoch precedente spesso NON è salvato: salvo la distribuzione corrente come riferimento e una stima via proposedKeys (se presente).")
    print("- Le etichette 'main nodeX' / 'overflow nodeX' appaiono quando gli indirizzi corrispondono alla CONFIG iniziale.")


if __name__ == "__main__":
    main()
