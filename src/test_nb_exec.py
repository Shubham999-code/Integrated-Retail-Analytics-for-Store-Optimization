import nbformat
from nbclient import NotebookClient

nb = nbformat.v4.new_notebook()
nb.cells.append(nbformat.v4.new_code_cell("import matplotlib.pyplot as plt\nplt.plot([1, 2], [3, 4])\nplt.title('Test Plot')\nplt.show()"))

client = NotebookClient(nb, timeout=60, kernel_name='python3')
client.execute()
print("Executed test cells! Outputs:", len(nb.cells[0].outputs))
for out in nb.cells[0].outputs:
    print("Output type:", out.output_type)
