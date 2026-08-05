# Activation-viz

Simple GUI App which allows you to visualize llm queries on a small local model.

- GUI written with Python + TKinter
- LLM: HuggingFaceTB/SmolLM-135M-Instruct
    - For now, the LLM used is not configurable

## Demo

https://github.com/user-attachments/assets/eacb7089-2e7a-42ac-abdb-674c469a28a4

## Quickstart

Requirements:
- Python 3.14+
- uv [(Download guide)](https://docs.astral.sh/uv/getting-started/installation/)

Note: running gui.py the first time might take a bit. The local LLM model files 
are downloaded from HuggingFace. All model files are cached for subsequent runs.

```sh
uv sync
uv run gui.py
```
