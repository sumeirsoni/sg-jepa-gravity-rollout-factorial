# Analysis files

`results.csv` contains the means across the five evaluation seeds for the four
factorial cells. Each row reports normalized target MSE and position L2 at one
evaluation horizon.

Regenerate the figures from the CSV with:

```bash
python analysis/plot_results.py
```

The script uses only the Python standard library and writes the two SVG files
under `figures/`.
