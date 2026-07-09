"""Run all four GAMES103 labs end-to-end and regenerate everything in results/.

    python run_all.py

Each lab is a self-contained NumPy simulation that writes rendered frames, GIFs,
verification plots and a *_metrics.json into results/.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LABS = [
    ("lab1_rigidbody", "run.py"),
    ("lab2_cloth", "run.py"),
    ("lab3_fem_elastic", "run.py"),
    ("lab4_shallow_wave", "run.py"),
]

for folder, script in LABS:
    path = os.path.join(HERE, folder, script)
    print(f"\n########## running {folder}/{script} ##########")
    sys.argv = [path]
    # run each lab from its own directory so relative imports resolve
    os.chdir(os.path.join(HERE, folder))
    runpy.run_path(path, run_name="__main__")

print("\nAll labs complete. See results/ for GIFs, plots and *_metrics.json.")
