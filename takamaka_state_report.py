#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Takamaka STATE report — INTEGRATED
KPI estratti da file .state (JSON):
- Per nodeHexSignerHash (da proposedKeys): blocks consegnati
- Per URL64 (da currentEpochSlotDistribution/operationalRecord):
  assigned_slots, delivered_blocks, efficiency, coinbase/fees/frozen, penaltySlots
In più:
- Scelta snapshot più recente in base a currentStateTime (fallback mtime)
- Name mapping opzionale (--name-map CSV con colonne URL64,label)
- KPI rate: penalty_rate_per100, fees_per_slot
- Aggregazioni per epoca se si passano più snapshot
Output: CSV, grafici PNG (etichette non tagliate), report HTML
"""

import argparse, json, gzip, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Plot style ----------
FIGSIZE = (10, 6)      # più ampio
FONTSIZE = 12
TICKSIZE = 10
TITLESIZE = 16

plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": TITLESIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": TICKSIZE,
    "ytick.labelsize": TICKSIZE,
})

def _abbr(s: str, keep: int = 6) -> str:
    if not isinstance(s, str): return str(s)
    return s if len(s) <= keep*2+1 else f"{s[:keep]}…{s[-keep:]}"

# ---------- IO helpers ----------

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    op = gzip.open if path.suffix == ".gz" else open
    try:
        with op(path, "rt", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return None

def collect_inputs(paths: List[str]) -> List[Path]:
    files: List[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(p.rglob("*.state"))
            files.extend(p.rglob("*.state.json"))
            files.extend(p.rglob("*.state.json.gz"))
    return sorted(files)

# ---------- Parsing ----------

def parse_proposed_keys(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    pk = data.get("proposedKeys", {})
    if isinstance(pk, dict):
        for name, rec in pk.items():
            rows.append({
                "proposedKeyName": name,
                "epoch": rec.get("epoch"),
                "slot": rec.get("slot"),
                "nodeHexSignerHash": rec.get("nodeHexSignerHash"),
                "nodePublicKey": rec.get("nodePublicKey"),
                "blockPublicKeyHash": rec.get("blockPublicKeyHash"),
                "burned": rec.get("burned", None),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df["slot"] = pd.to_numeric(df["slot"], errors="coerce")
        df = df[(df["burned"].isna()) | (df["burned"] == False)]
        df = df.sort_values(["epoch","slot"], na_position="last")
    return df

def parse_slot_distribution(data: Dict[str, Any]) -> pd.DataFrame:
    d = data.get("currentEpochSlotDistribution", {})
    rows = []
    if isinstance(d, dict):
        for url64, slots in d.items():
            n = len(slots) if isinstance(slots, list) else 0
            rows.append({"URL64": url64, "assigned_slots": n})
    df = pd.DataFrame(rows)
    return df.sort_values("assigned_slots", ascending=False) if not df.empty else df

def parse_operational_record(data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rec = data.get("operationalRecord", {})
    main_rows, deliv_rows = [], []
    if isinstance(rec, dict):
        for _, obj in rec.items():
            um = obj.get("urlMainRecords", {})
            if isinstance(um, dict):
                main_rows.append({
                    "URL64": um.get("mainUrl64Addr"),
                    "holderCoinbase": um.get("holderCoinbase", 0),
                    "nodeCoinbase": um.get("nodeCoinbase", 0),
                    "holderRedFee": um.get("holderRedFee", 0),
                    "holderGreenFee": um.get("holderGreenFee", 0),
                    "nodeRedFee": um.get("nodeRedFee", 0),
                    "nodeGreenFee": um.get("nodeGreenFee", 0),
                    "holderRedFrozenFee": um.get("holderRedFrozenFee", 0),
                    "holderGreenFrozenFee": um.get("holderGreenFrozenFee", 0),
                    "nodeRedFrozenFee": um.get("nodeRedFrozenFee", 0),
                    "nodeGreenFrozenFee": um.get("nodeGreenFrozenFee", 0),
                    "penaltySlots": um.get("penaltySlots", 0),
                })
            uc = obj.get("urlOverflowCounter", {})
            if isinstance(uc, dict):
                deliv_rows.append({
                    "overflowUrl64Addr": uc.get("overflowUrl64Addr"),
                    "mainUrl64Addr": uc.get("mainUrl64Addr"),
                    "deliveredBlocks": uc.get("deliveredBlocks", 0),
                })
    main_df = pd.DataFrame(main_rows)
    deliv_df = pd.DataFrame(deliv_rows)
    if not deliv_df.empty:
        deliv_df["deliveredBlocks"] = pd.to_numeric(deliv_df["deliveredBlocks"], errors="coerce").fillna(0).astype(int)
    return main_df, deliv_df

# ---------- KPI ----------

def kpi_by_nodehash(df_pk: pd.DataFrame) -> pd.DataFrame:
    if df_pk.empty:
        return pd.DataFrame(columns=["nodeHexSignerHash","blocks_delivered"])
    s = df_pk.groupby("nodeHexSignerHash").size().rename("blocks_delivered").sort_values(ascending=False)
    return s.reset_index()

def kpi_by_url64(dist_df: pd.DataFrame, deliv_df: pd.DataFrame, main_df: pd.DataFrame) -> pd.DataFrame:
    # base: assigned
    base = dist_df.copy() if not dist_df.empty else pd.DataFrame(columns=["URL64","assigned_slots"])
    # delivered
    if not deliv_df.empty and deliv_df["mainUrl64Addr"].notna().any():
        delivered = deliv_df.groupby("mainUrl64Addr")["deliveredBlocks"].sum().rename_axis("URL64").rename("delivered_blocks").reset_index()
    elif not deliv_df.empty:
        delivered = deliv_df.groupby("overflowUrl64Addr")["deliveredBlocks"].sum().rename_axis("URL64").rename("delivered_blocks").reset_index()
    else:
        delivered = pd.DataFrame(columns=["URL64","delivered_blocks"])
    # rewards
    if not main_df.empty:
        rewards = (main_df.groupby("URL64", dropna=True).agg({
            "holderCoinbase":"sum","nodeCoinbase":"sum",
            "holderRedFee":"sum","holderGreenFee":"sum",
            "nodeRedFee":"sum","nodeGreenFee":"sum",
            "holderRedFrozenFee":"sum","holderGreenFrozenFee":"sum",
            "nodeRedFrozenFee":"sum","nodeGreenFrozenFee":"sum",
            "penaltySlots":"sum"}).reset_index())
    else:
        rewards = pd.DataFrame(columns=["URL64"])

    out = base.merge(delivered, on="URL64", how="outer").merge(rewards, on="URL64", how="left")
    for c in ["assigned_slots","delivered_blocks"]:
        if c not in out: out[c] = 0
        out[c] = out[c].fillna(0).astype(int)
    out["efficiency"] = (out["delivered_blocks"] / out["assigned_slots"]).where(out["assigned_slots"] > 0)

    for c in ["holderCoinbase","nodeCoinbase","holderRedFee","holderGreenFee","nodeRedFee","nodeGreenFee",
              "holderRedFrozenFee","holderGreenFrozenFee","nodeRedFrozenFee","nodeGreenFrozenFee","penaltySlots"]:
        if c not in out: out[c] = 0
        out[c] = out[c].fillna(0)
    out["total_coinbase"] = out["holderCoinbase"] + out["nodeCoinbase"]
    out["total_fees"] = out["holderRedFee"] + out["holderGreenFee"] + out["nodeRedFee"] + out["nodeGreenFee"]
    out["total_frozen_fees"] = out["holderRedFrozenFee"] + out["holderGreenFrozenFee"] + out["nodeRedFrozenFee"] + out["nodeGreenFrozenFee"]

    # KPI rate aggiuntivi
    eps = out["assigned_slots"].replace(0, pd.NA)
    out["penalty_rate_per100"] = (out["penaltySlots"] / eps * 100)
    out["fees_per_slot"] = (out["total_fees"] / eps)

    return out.sort_values(["assigned_slots","delivered_blocks"], ascending=[False, False])

# ---------- Plot helpers (etichette non tagliate) ----------

def savefig_tight(path: Path):
    plt.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.25)
    plt.close()

def plot_bar(series: pd.Series, title: str, outpath: Path):
    if series.empty: return
    fig, ax = plt.subplots(figsize=FIGSIZE)
    s = series.copy()
    s.index = [_abbr(str(x)) for x in s.index]  # abbrevia etichette
    s.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Count" if "rate" not in title.lower() and "fees" not in title.lower() else "")
    ax.set_xlabel(s.index.name if s.index.name else "")
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    savefig_tight(outpath)

def plot_pie(series: pd.Series, title: str, outpath: Path, topn: int = 12):
    if series.empty: return
    s = series.sort_values(ascending=False)
    other = s[topn:].sum()
    pie = s.head(topn).copy()
    if other > 0: pie.loc["Others"] = other
    labels = [_abbr(str(x)) for x in pie.index]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.pie(pie.values, autopct="%1.1f%%")
    ax.set_title(title)
    # legenda a destra per non sovrapporre le etichette
    ax.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    savefig_tight(outpath)

# ---------- Report HTML ----------

def write_report_html(outdir: Path, inputs: List[Path], meta: Dict[str, Any]):
    html = []
    html.append("<html><head><meta charset='utf-8'><title>Takamaka State Report</title></head><body>")
    html.append("<h1>Takamaka State Report</h1>")
    html.append("<h3>Input files</h3><ul>")
    for p in inputs: html.append(f"<li>{p}</li>")
    html.append("</ul>")
    if meta:
        html.append("<h3>Metadata</h3><ul>")
        for k,v in meta.items(): html.append(f"<li><b>{k}</b>: {v}</li>")
        html.append("</ul>")
    html.append("<h3>Outputs</h3><ul>")
    for fn in ["by_nodehash.csv","by_url64.csv","by_url64_per_epoch.csv","assigned_slots.csv","delivered_blocks_url64.csv","rewards_url64.csv","summary_topn.csv",
               "plot_nodehash_blocks_bar.png","plot_nodehash_blocks_pie.png",
               "plot_url64_assigned_bar.png","plot_url64_assigned_pie.png",
               "plot_url64_delivered_bar.png","plot_url64_efficiency_bar.png",
               "plot_url64_rewards_coinbase_bar.png","plot_url64_fees_total_bar.png",
               "plot_url64_penalty_rate_bar.png","plot_url64_fees_per_slot_bar.png"]:
        if (outdir / fn).exists():
            html.append(f"<li><a href='{fn}'>{fn}</a></li>")
    html.append("</ul>")
    for fn in ["plot_nodehash_blocks_bar.png","plot_nodehash_blocks_pie.png",
               "plot_url64_assigned_bar.png","plot_url64_assigned_pie.png",
               "plot_url64_delivered_bar.png","plot_url64_efficiency_bar.png",
               "plot_url64_rewards_coinbase_bar.png","plot_url64_fees_total_bar.png",
               "plot_url64_penalty_rate_bar.png","plot_url64_fees_per_slot_bar.png"]:
        if (outdir / fn).exists():
            html.append(f"<h3>{fn}</h3><img src='{fn}' style='max-width:1100px;'>")
    html.append("</body></html>")
    (outdir / "report.html").write_text("\n".join(html), encoding="utf-8")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input","-i", required=True, nargs="+", help="File o directory .state (JSON)")
    ap.add_argument("--output","-o", required=True, help="Cartella output del report")
    ap.add_argument("--topn", type=int, default=15, help="Top N per i grafici")
    ap.add_argument("--timestamped", action="store_true", help="Crea sottocartella con timestamp")
    ap.add_argument("--name-map", help="CSV con colonne URL64,label per rinominare nodi", default=None)
    args = ap.parse_args()

    inputs = collect_inputs(args.input)
    if not inputs:
        print("[ERR] Nessun file trovato.")
        sys.exit(2)

    outdir = Path(args.output)
    if args.timestamped:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        outdir = outdir / f"takamaka_report_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # Carica tutti i JSON validi e scegli il più recente per meta/distribuzioni
    candidates: List[Tuple[int, Path, Dict[str, Any]]] = []
    df_pk_all = []
    for p in inputs:
        d = read_json(p)
        if not d:
            continue
        ts = d.get("currentStateTime") or 0
        try:
            ts = int(ts)
        except Exception:
            ts = 0
        candidates.append((ts, p, d))
        df_pk_all.append(parse_proposed_keys(d))

    if not candidates:
        print("[ERR] Nessun JSON valido.")
        sys.exit(3)

    candidates.sort(key=lambda x: (x[0], x[1].stat().st_mtime))
    latest_ts, latest_path, latest_data = candidates[-1]

    df_pk = pd.concat(df_pk_all, ignore_index=True) if df_pk_all else pd.DataFrame()

    # Meta
    meta = {}
    if latest_data:
        cs = latest_data.get("currentStateTime")
        wi = latest_data.get("worldInitTime")
        meta = {
            "epoch": latest_data.get("epoch"),
            "slot": latest_data.get("slot"),
            "currentStateTime": pd.to_datetime(cs, unit="ms", utc=True) if cs else None,
            "worldInitTime": pd.to_datetime(wi, unit="ms", utc=True) if wi else None,
            "chainWeight": latest_data.get("chainWeight"),
            "currentEpochSlotWeight": latest_data.get("currentEpochSlotWeight"),
            "input_used_for_meta": str(latest_path) if latest_path else None
        }

    # Distribuzione/Rewards/Delivered dal più recente
    dist_df = parse_slot_distribution(latest_data or {})
    main_df, deliv_df = parse_operational_record(latest_data or {})

    # KPI
    by_node = kpi_by_nodehash(df_pk)
    by_url = kpi_by_url64(dist_df, deliv_df, main_df)

    # Name map opzionale
    name_map = {}
    if args.name_map:
        try:
            nm = pd.read_csv(args.name_map)
            if {"URL64","label"} <= set(nm.columns):
                name_map = dict(zip(nm["URL64"], nm["label"]))
        except Exception:
            name_map = {}

    def _apply_labels(s: pd.Series) -> pd.Series:
        if not name_map:
            return s
        s = s.copy()
        s.index = [name_map.get(i, i) for i in s.index]
        return s

    # CSV principali
    by_node.to_csv(outdir / "by_nodehash.csv", index=False)
    by_url.to_csv(outdir / "by_url64.csv", index=False)
    dist_df.to_csv(outdir / "assigned_slots.csv", index=False)
    deliv_df.to_csv(outdir / "delivered_blocks_url64.csv", index=False)
    main_df.to_csv(outdir / "rewards_url64.csv", index=False)

    # Per-epoca se input multipli
    per_epoch = []
    for ts, p, d in candidates:
        e = d.get("epoch")
        dist = parse_slot_distribution(d)
        mdf, ddf = parse_operational_record(d)
        k = kpi_by_url64(dist, ddf, mdf)
        if not k.empty:
            tmp = k.loc[:, ["URL64","assigned_slots","delivered_blocks","efficiency"]].copy()
            tmp["epoch"] = e
            per_epoch.append(tmp)
    df_epoch = pd.concat(per_epoch, ignore_index=True) if per_epoch else pd.DataFrame()
    if not df_epoch.empty:
        df_epoch.to_csv(outdir / "by_url64_per_epoch.csv", index=False)

    # Tabellina riassuntiva TopN
    if not by_url.empty:
        cols = ["URL64","assigned_slots","delivered_blocks","efficiency","total_coinbase","total_fees","penalty_rate_per100","fees_per_slot"]
        cols = [c for c in cols if c in by_url.columns]
        summary_topn = by_url.loc[:, cols].head(min(args.topn, len(by_url)))
        summary_topn.to_csv(outdir / "summary_topn.csv", index=False)

    # Grafici (etichette pulite + label mapping)
    if not by_node.empty:
        s_blocks = by_node.set_index("nodeHexSignerHash")["blocks_delivered"]
        plot_bar(s_blocks.head(args.topn), "Blocks per nodeHexSignerHash (Top N)", outdir / "plot_nodehash_blocks_bar.png")
        plot_pie(s_blocks, "Share of blocks by nodeHexSignerHash", outdir / "plot_nodehash_blocks_pie.png", topn=args.topn)

    if not by_url.empty:
        s_assigned = by_url.set_index("URL64")["assigned_slots"]
        s_delivered = by_url.set_index("URL64")["delivered_blocks"]
        s_eff = by_url.set_index("URL64")["efficiency"].dropna()

        s_assigned = _apply_labels(s_assigned)
        s_delivered = _apply_labels(s_delivered)
        s_eff = _apply_labels(s_eff)

        plot_bar(s_assigned.head(args.topn), "Assigned slots by URL64 (Top N)", outdir / "plot_url64_assigned_bar.png")
        plot_pie(s_assigned, "Share of assigned slots by URL64", outdir / "plot_url64_assigned_pie.png", topn=args.topn)
        plot_bar(s_delivered.head(args.topn), "Delivered blocks by URL64 (Top N)", outdir / "plot_url64_delivered_bar.png")
        plot_bar(s_eff.sort_values(ascending=False).head(args.topn), "Efficiency (delivered/assigned) by URL64 (Top N)", outdir / "plot_url64_efficiency_bar.png")

        if "total_coinbase" in by_url.columns:
            s_coin = _apply_labels(by_url.set_index("URL64")["total_coinbase"])
            plot_bar(s_coin.head(args.topn), "Total coinbase by URL64 (Top N)", outdir / "plot_url64_rewards_coinbase_bar.png")
        if "total_fees" in by_url.columns:
            s_fees = _apply_labels(by_url.set_index("URL64")["total_fees"])
            plot_bar(s_fees.head(args.topn), "Total fees (red+green) by URL64 (Top N)", outdir / "plot_url64_fees_total_bar.png")
        if "penalty_rate_per100" in by_url.columns:
            s_pen = _apply_labels(by_url.set_index("URL64")["penalty_rate_per100"].fillna(0))
            plot_bar(s_pen.sort_values(ascending=False).head(args.topn), "Penalty rate per 100 slots by URL64 (Top N)", outdir / "plot_url64_penalty_rate_bar.png")
        if "fees_per_slot" in by_url.columns:
            s_fps = _apply_labels(by_url.set_index("URL64")["fees_per_slot"].fillna(0))
            plot_bar(s_fps.sort_values(ascending=False).head(args.topn), "Fees per slot by URL64 (Top N)", outdir / "plot_url64_fees_per_slot_bar.png")

    write_report_html(outdir, inputs, meta)
    print(f"[OK] Report in: {outdir.resolve()}")
    sys.exit(0)

if __name__ == "__main__":
    main()