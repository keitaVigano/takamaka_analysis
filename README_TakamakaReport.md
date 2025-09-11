# Takamaka STATE report — INTEGRATED

## Come usarlo

```bash
python3 takamaka_state_report.py   -i /path/ai/tuoi/*.state /anche/cartelle   -o ./out_report   --timestamped   --topn 15   --name-map ./example_name_map.csv
```

- `-i/--input`: uno o più file o directory contenenti `.state`, `.state.json`, `.state.json.gz`.
- `-o/--output`: cartella di destinazione.
- `--timestamped`: crea una sottocartella con timestamp.
- `--topn`: quanti elementi mostrare nei grafici.
- `--name-map`: CSV opzionale con colonne `URL64,label` per sostituire gli hash con etichette leggibili.

## Output principali
- `by_nodehash.csv`: blocchi per `nodeHexSignerHash` (da `proposedKeys` su tutti gli snapshot trovati).
- `by_url64.csv`: KPI per `URL64` dallo snapshot più recente (slot assegnati, blocchi consegnati, efficienza, coinbase/fees/frozen, penaltySlots, rate).
- `by_url64_per_epoch.csv`: (se disponibili più snapshot) KPI per epoca.
- Grafici PNG con etichette non tagliate.
- `report.html` con link a tutti gli output.

