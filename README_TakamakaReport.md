# Takamaka STATE Report — Integrated

CLI tool to analyze one or more **Takamaka state snapshots** and generate tables & charts with KPIs for **main/overflow** and **node hashes**. Works with single files or directories.

---

## Requirements

- Python **3.9+**
- Packages: `pandas`, `matplotlib`

~~~bash
pip install pandas matplotlib
~~~

---

## Usage

~~~bash
python3 takamaka_state_report.py \
  -i /path/to/file1.state /path/to/file2.state /path/to/dir_with_states \
  -o ./out_report \
  --timestamped \
  --topn 15 \
  --name-map ./example_name_map.csv
~~~

### Arguments

- `-i, --input`
  One or more **files** or **directories** containing:
  - `.state`
  - `.state.json`
  - `.state.json.gz`

- `-o, --output`
  Output directory (e.g., `./out_report`).

- `--timestamped` *(optional)*
  Create a **timestamped** subfolder inside the output dir.
  Example: `out_report/2025-09-11_19-25-03/…`

- `--topn <int>` *(optional, default: 10)*
  Number of items to display in charts (sorted by the relevant metric).

- `--name-map <file.csv>` *(optional)*
  CSV with columns `URL64,label` to replace Base64 IDs (`mainUrl64Addr` / `lowUrl64Addr`) with readable labels in CSVs and charts.

---

## Outputs

All files are saved under `OUTPUT_ROOT/(timestamp)/` (if `--timestamped`) or directly in `OUTPUT_ROOT/`.

### CSVs

- **`by_nodehash.csv`**
  Aggregation by `nodeHexSignerHash` (from `proposedKeys`) across **all found snapshots**. Useful to see how many **proposed slots/blocks** per node hash.

- **`by_url64.csv`**
  KPIs by `URL64` (main/overflow) from the **most recent snapshot**:
  - **assigned slots** (when available),
  - **delivered/missed blocks** (overflow),
  - **efficiency** (e.g., `delivered / assigned` when meaningful),
  - **coinbase/fees/frozen** (node/holder) from `urlMainRecords`,
  - **penaltySlots** and **rates** (e.g., penalties/assigned).

- **`by_url64_per_epoch.csv`** *(when multiple snapshots with different epochs exist)*
  KPIs **per epoch**, by `URL64`, to track evolution over time.

> When a metric isn’t natively available (e.g., assigned slots for past epochs), the tool may **estimate** it from `proposedKeys`. Estimated fields are marked accordingly.

### Charts (PNG)

- Top **main/overflow** by key KPIs (based on `--topn`), with readable labels. Examples:
  - `penalty_slots_per_main.png`
  - `node_rewards_per_main.png`
  - `delivered_blocks_per_overflow.png`
  - `stake_slots_reference.png` (if current distribution is available)

### HTML Report

- **`report.html`**
  Index with links to **all outputs** (CSVs + PNGs) for quick navigation.

---

## Optional name-map CSV

~~~csv
URL64,label
Be2ntZsJLQ0Kmp_KRv-281k-x5pWqnuUNNzFqsi35lQ.,main node1
_0OHvruLBmlRlDnCEmKdxyc65hBynAxRtiuQ8BHkrkM.,overflow node1
Lb5sLN6alt5KcdMNE74onoOuQL8Y0eYgGZKqDOAUSTU.,main node3
~~~

- `URL64` must **exactly match** addresses in the dump (`mainUrl64Addr` / `lowUrl64Addr`).
- `label` is the text shown in CSVs and charts.

---

## Metrics (what it measures)

- **Assigned slots**
  - **Current epoch**: from `currentEpochSlotDistribution` (if present).
  - **Past epochs**: when not explicitly stored, may be **estimated** from `proposedKeys`.

- **Delivered/missed blocks** (overflow)
  - From `urlOverflowCounter.deliveredBlocks` / `missedBlocks` per epoch.

- **Rewards & fees** (main)
  - From `urlMainRecords`:
    - `nodeCoinbase`, `holderCoinbase`
    - `nodeRedFee`, `holderRedFee`, `nodeGreenFee`, `holderGreenFee`
    - Frozen: `nodeRedFrozenFee`, `holderRedFrozenFee`, `nodeGreenFrozenFee`, `holderGreenFrozenFee`

- **Penalties**
  - `penaltySlots` for main and related rates (e.g., penalties/assigned).

---

## Examples

1) Report on a directory of many `.state` files:
~~~bash
python3 takamaka_state_report.py -i ./dumps -o ./out_report
~~~

2) With label mapping and top-20 charts:
~~~bash
python3 takamaka_state_report.py \
  -i ./dumps ./other/state.json.gz \
  -o ./out_report \
  --timestamped \
  --topn 20 \
  --name-map ./names.csv
~~~

---

## Notes & Limitations

- Snapshots **do not always** contain full historical data (e.g., assigned slots for past epochs). In such cases:
  - the **current distribution** is saved as a **reference**,
  - and a **best-effort estimate** is derived from `proposedKeys` (partial if the snapshot is mid-epoch).
- If you have **multiple snapshots** for the same epoch, the tool keeps the **latest** KPIs for that epoch.
- If a row is not found in `--name-map`, the original `URL64` is used.

---

## Troubleshooting

- **Empty or partial outputs**
  Ensure the files include expected sections (`operationalRecord`, `proposedKeys`, `currentEpochSlotDistribution`).

- **Charts with few bars**
  Increase `--topn` or verify the KPIs actually exist in the snapshots.

- **Labels not replaced**
  Confirm the `URL64` column in your `--name-map` matches the dump addresses **exactly**.

---

## License

**GNU General Public License v3.0 (GPL-3.0)** — Version 3, 29 June 2007.
Include the full license text in a `LICENSE` file in your repository.
