# Takamaka Analysis — Dashboard and PNG Export

Tools to analyze a **Takamaka** state snapshot and produce charts for key metrics across main and overflow nodes.

Two usage modes are included:

- **PNG export** from the command line via `takamaka_dash.py`
- **Interactive dashboard** (tabs) via `analyze_takamaka.py`

---

## Requirements

- Python **3.9+**
- Packages:
  - `numpy`
  - `plotly`
  - `matplotlib` (for color conversions only)
  - `kaleido` (required to save PNGs with Plotly)
  - `dash` (only for the interactive dashboard)

Quick install:

```bash
pip install numpy plotly matplotlib kaleido dash
```

---

## Input data

Both scripts read a JSON state file (e.g., `three.state`) with fields such as `epoch`, `operationalRecord`, `currentEpochSlotDistribution`, etc.

At the top of the code you can configure these core parameters:

- `FILENAME`: path to the `.state` file
- `OUTPUT_DIR`: output folder for PNGs
- `MAX_SLOT`: last slot considered (e.g., 2399)
- `LABEL`: map `URL64 -> readable label`
- `MAIN_ADDRS`, `OVERFLOW_ADDRS`: canonical order of addresses to display
- `STAKE_EPOCH`, `STAKE_SLOT`, `STAKE_RAW`: stake values for main nodes (optional, used by the stake chart)

---

## Mode 1 — PNG export (takamaka_dash.py)

Generates three static PNG charts in `OUTPUT_DIR`:

- Stake per main node (uses `STAKE_*` and `STAKE_RAW`)
- Penalty slots per main node (from `operationalRecord` at the current epoch)
- Overflow summary: bars for delivered blocks + area with cumulative share of validated slots

Example:

```bash
python takamaka_dash.py
```

Expected outputs (indicative names):

- `output/stake_epoch_<E>_slot_<S>.png`
- `output/penalty_slots_epoch_<E>.png`
- `output/summary_epoch_<E>.png`

The repo includes sample images in `output/` (e.g., `penalty_slots_epoch_2.png`, `summary_epoch_3.png`).

Note: PNG export requires `kaleido` to be installed.

---

## Mode 2 — Interactive dashboard (analyze_takamaka.py)

Starts a **Dash** app with three tabs: Stake, Penalty Slots, Summary. These are the same charts as PNG mode but browsable in your browser.

Run:

```bash
python analyze_takamaka.py
```

By default it runs with `debug=True` at `http://127.0.0.1:8050/`.

---

## Metric details

- **Stake (main)**: values taken from `STAKE_RAW` and shown for addresses in `MAIN_ADDRS`
- **Penalty slots (main)**: from `operationalRecord[epoch].urlMainRecords[addr].penaltySlots`
- **Delivered blocks (overflow)**: from `operationalRecord[epoch].urlOverflowCounter[addr].deliveredBlocks`
- **Validated slots share (overflow)**: computed over `currentEpochSlotDistribution` up to `MAX_SLOT`

If a metric or address is missing from the data, it is shown as 0.

---

## Tips and troubleshooting

- If PNGs are not saved, ensure `kaleido` is installed.
- If charts look empty or partial, check that the `.state` file contains `operationalRecord` and (for the summary) `currentEpochSlotDistribution`.
- Update `LABEL`, `MAIN_ADDRS` and `OVERFLOW_ADDRS` to match the real addresses in your scenario.

---

## License

This project is released under **GNU General Public License v3.0 (GPL-3.0)**. The full text is in `LICENSE`.