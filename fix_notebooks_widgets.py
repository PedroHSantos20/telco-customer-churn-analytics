import nbformat
from pathlib import Path

# Folder where your notebooks live
notebooks_dir = Path("notebooks")

for nb_path in notebooks_dir.glob("*.ipynb"):
    print(f"Processing {nb_path}...")
    nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)

    # If the top-level metadata has a "widgets" key, remove it
    if "widgets" in nb.metadata:
        print("  - Removing metadata.widgets")
        nb.metadata.pop("widgets")

    nbformat.write(nb, nb_path)

print("Done. Notebooks cleaned.")
