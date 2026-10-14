import json

with open('Sample_ML_Submission_Template.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
with open('src/template_outline.txt', 'w', encoding='utf-8') as out:
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        ctype = c['cell_type']
        first_line = src.strip().split('\n')[0] if src.strip() else '(empty)'
        out.write(f"Cell {i:3d} [{ctype:4s}]: {first_line[:100]}\n")

print("Outline written to src/template_outline.txt")
