# Publish GuppyLM to Ollama (Portable, Standard App)

This workflow is code-only (no pre-generated binaries committed) and works from a fresh clone.

## One Command

From repo root:

```bash
python tools/publish_ollama.py
```

On Windows:

```powershell
py tools\publish_ollama.py
```

## What the Script Does

1. Runs preflight status checks.
2. Finds or creates a virtual environment (asks path if needed).
3. Installs required Python packages in that environment.
4. Downloads required GuppyLM model files from Hugging Face if missing.
5. Builds canonical tensors and exports a portable GGUF (`gpt2` arch).
6. Creates a local Ollama model and runs a smoke test.
7. Checks Ollama signin (runs `ollama signin` if needed).
8. Asks for namespace and model name (default remote model: `guppylm`).
9. Pushes to Ollama and prints the final URL.

## Notes

- `llama.cpp` is not required for this portable path.
- Portable model uses a stock Ollama-compatible architecture so standard Ollama app can run it.
- If Ollama server is not running, start Ollama app (or `ollama serve`) and rerun.
