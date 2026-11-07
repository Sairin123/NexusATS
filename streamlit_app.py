"""Streamlit Community Cloud standard entrypoint."""
import runpy
from pathlib import Path

if __name__ == "__main__" or True:
    runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")
