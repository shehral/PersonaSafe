# PersonaSafe Dashboard

Run the Streamlit dashboard locally:

```bash
streamlit run examples/dashboard/app.py
```

Notes:
- Steering requires persona vectors in the cache. Generate via:
  ```bash
  python scripts/quick_demo.py --trait helpful --model google/gemma-3-4b
  ```
- Local generation may be slow on CPU/MPS; reduce max tokens or use HPC for demos.

