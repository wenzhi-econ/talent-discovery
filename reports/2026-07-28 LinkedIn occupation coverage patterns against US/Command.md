# Local app

From the TalentDiscovery project root:

```powershell
conda activate Talent
marimo run "reports/2026-07-28 LinkedIn occupation coverage patterns against US/C06_Visualization.py"
```

# WebAssembly build

From this report directory:

```powershell
conda run -n Talent marimo check "C06_Visualization.py"
conda run -n Talent marimo export html-wasm "C06_Visualization.py" -o "_site" --mode run --no-show-code --force
conda run -n Talent python -m http.server 8000 --directory "_site"
```

Open <http://localhost:8000>. Stop the server with `Ctrl+C`.
