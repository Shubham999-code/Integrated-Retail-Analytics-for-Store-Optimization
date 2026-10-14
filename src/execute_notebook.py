"""
execute_notebook.py
-------------------
Executes all cells in Integrated_Retail_Analytics_Store_Optimization.ipynb
using nbclient and ipykernel, capturing all outputs, figures, and tables,
and writing back the fully executed notebook.
"""

import nbformat
from nbclient import NotebookClient
import time
import sys

print("Loading Integrated_Retail_Analytics_Store_Optimization.ipynb...", flush=True)
with open('Integrated_Retail_Analytics_Store_Optimization.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print(f"Total cells to execute: {len(nb.cells)}", flush=True)

client = NotebookClient(
    nb,
    timeout=600,
    kernel_name='python3',
    allow_errors=False
)

t0 = time.time()
print("Starting execution across all cells...", flush=True)
client.execute()
print(f"Execution finished in {time.time()-t0:.1f} seconds!", flush=True)

with open('Integrated_Retail_Analytics_Store_Optimization.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Saved fully executed notebook to Integrated_Retail_Analytics_Store_Optimization.ipynb!", flush=True)

# Count executed outputs
total_outputs = sum(len(c.get('outputs', [])) for c in nb.cells if c.cell_type == 'code')
print(f"Total code cell outputs embedded: {total_outputs}", flush=True)
