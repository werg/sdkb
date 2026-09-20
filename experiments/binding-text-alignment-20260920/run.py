"""Run matched alignment/control training and frozen stored-only confirmations."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).parents[1]/'binding-reader-capacity-20260920/run.py'),
                   run_name='__main__')
