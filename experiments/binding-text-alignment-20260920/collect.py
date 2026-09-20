"""Collect the complete declared latent and selected-text alignment confirmations."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).parents[1]/'binding-reader-capacity-20260920/collect.py'),
                   run_name='__main__')
