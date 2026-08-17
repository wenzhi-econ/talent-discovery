# LinkedIn occupation coverage in 2022

This folder contains a self-contained marimo application comparing US LinkedIn
occupation and age distributions with ACS, CPS, and OEWS benchmarks.

## Run locally

From the TalentDiscovery project root:

```powershell
conda activate Talent
marimo run "reports/2026-07-28 LinkedIn occupation coverage patterns against US/C06_Visualization.py"
```

## Build and preview the website

From this report directory:

```powershell
conda run -n Talent marimo check "C06_Visualization.py"
conda run -n Talent marimo export html-wasm "C06_Visualization.py" -o "_site" --mode run --no-show-code --force
conda run -n Talent python -m http.server 8000 --directory "_site"
```

Visit <http://localhost:8000> and stop the server with `Ctrl+C`.

## Publication warning

Everything under `public/` is copied into the website and is downloadable by
visitors. Public deployment therefore requires approval under the Revelio data
licence and any applicable disclosure-control rules.
