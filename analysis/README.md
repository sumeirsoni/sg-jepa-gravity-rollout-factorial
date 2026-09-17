# Analysis files

`results.csv` contains the means across the five evaluation seeds for the four
factorial cells. Each row reports normalized target MSE and position L2 at one
evaluation horizon.

Regenerate the figures from the CSV with Matplotlib:

```bash
uv sync --extra analysis
uv run --extra analysis python analysis/plot_results.py
```

The script writes SVG and PDF vector figures plus high-resolution PNG previews
under `figures/`. The SVG files are embedded in the main README.
