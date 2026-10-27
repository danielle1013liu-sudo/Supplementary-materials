"""Streamlit Community Cloud entrypoint."""

from pathlib import Path
import runpy
import sys

repository = Path(__file__).resolve().parents[1]
if str(repository) not in sys.path:
    sys.path.insert(0, str(repository))
runpy.run_path(str(repository / 'app.py'), run_name='__main__')
