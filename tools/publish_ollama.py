#!/usr/bin/env python3
"""Build and push a GuppyLM Ollama model from a fresh clone.

This script performs interactive preflight checks, downloads required model files,
exports GGUF, creates an Ollama model, runs a smoke test, and optionally pushes
to ollama.com under <username>/<model>.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import textwrap
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO_ID = "arman-bd/guppylm-9M"
HF_MODEL_DIR = REPO_ROOT / "data" / "hf" / "guppylm-9M"
GGUF_PREP_DIR = REPO_ROOT / "artifacts" / "guppylm_gguf_prep"
PORTABLE_GGUF = GGUF_PREP_DIR / "guppylm-gpt2-f32.gguf"
MODELSPEC_PATH = REPO_ROOT / "artifacts" / "ollama" / "Modelfile.guppylm"

REQUIRED_MODEL_FILES = [
    "pytorch_model.bin",
    "config.json",
    "tokenizer.json",
]

REQUIRED_PY_PACKAGES = [
    "torch",
    "numpy",
    "gguf",
    "tokenizers",
    "huggingface_hub",
]


def info(msg: str) -> None:
    print(f"[info] {msg}")


def warn(msg: str) -> None:
    print(f"[warn] {msg}")


def fail(msg: str, exit_code: int = 1) -> None:
    print(f"[error] {msg}")
    raise SystemExit(exit_code)


def prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value if value else (default or "")


def yes_no(text: str, default_yes: bool = True) -> bool:
    default_hint = "Y/n" if default_yes else "y/N"
    value = input(f"{text} ({default_hint}): ").strip().lower()
    if not value:
        return default_yes
    return value in {"y", "yes"}


def run(cmd: list[str], *, check: bool = True, capture: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess:
    kwargs = {
        "cwd": str(cwd) if cwd else None,
        "text": True,
    }
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.STDOUT
    proc = subprocess.run(cmd, **kwargs)
    if check and proc.returncode != 0:
        if capture and proc.stdout:
            print(proc.stdout)
        fail(f"Command failed: {' '.join(cmd)}")
    return proc


def detect_env_python(env_dir: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def ensure_venv() -> Path:
    default_env = REPO_ROOT / ".venv"
    use_existing = default_env.exists() and yes_no("Found .venv. Use it?", default_yes=True)
    if use_existing:
        py = detect_env_python(default_env)
        if py.exists():
            return py

    env_path_input = prompt("Python env path (new or existing)", str(default_env))
    env_dir = Path(env_path_input).expanduser().resolve()
    py = detect_env_python(env_dir)

    if py.exists():
        info(f"Using existing env: {env_dir}")
        return py

    if not yes_no(f"Env not found at {env_dir}. Create it now?", default_yes=True):
        fail("Python env is required.")

    info(f"Creating virtual environment at {env_dir}")
    env_dir.mkdir(parents=True, exist_ok=True)
    venv.create(str(env_dir), with_pip=True)
    py = detect_env_python(env_dir)
    if not py.exists():
        fail(f"Virtualenv created but Python not found at {py}")
    return py


def ensure_python_packages(py: Path) -> None:
    info("Upgrading pip in selected env")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    info("Installing required Python packages")
    run([str(py), "-m", "pip", "install", *REQUIRED_PY_PACKAGES], check=True)


def ensure_model_assets(py: Path) -> None:
    missing = [name for name in REQUIRED_MODEL_FILES if not (HF_MODEL_DIR / name).exists()]
    if not missing:
        info("Model assets already present")
        return

    info(f"Missing model assets: {', '.join(missing)}")
    info(f"Downloading from Hugging Face: {HF_REPO_ID}")

    script = textwrap.dedent(
        f"""
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id={HF_REPO_ID!r},
            local_dir={str(HF_MODEL_DIR)!r},
            local_dir_use_symlinks=False,
            allow_patterns={REQUIRED_MODEL_FILES!r},
        )
        """
    )
    run([str(py), "-c", script], check=True)

    still_missing = [name for name in REQUIRED_MODEL_FILES if not (HF_MODEL_DIR / name).exists()]
    if still_missing:
        fail(f"Download incomplete. Still missing: {', '.join(still_missing)}")


def ensure_ollama_ready() -> None:
    if shutil.which("ollama") is None:
        fail("`ollama` CLI is not found in PATH. Install Ollama first.")

    for attempt in range(3):
        proc = run(["ollama", "list"], check=False, capture=True)
        if proc.returncode == 0:
            return
        warn("Ollama server is not reachable.")
        if attempt < 2 and yes_no("Start Ollama and retry?", default_yes=True):
            continue
        fail("Could not reach Ollama server. Please run `ollama serve` or open Ollama app.")


def ensure_ollama_login() -> None:
    proc = run(["ollama", "signin"], check=False, capture=True)
    output = (proc.stdout or "").strip()
    user = parse_ollama_user(output)

    if proc.returncode == 0 and user:
        if output:
            info(output)
        return

    warn("Ollama signin could not be confirmed automatically.")
    if output:
        print(output)
    fail("Please run `ollama signin` manually, then rerun the script.")


def parse_ollama_user(output: str) -> str:
    patterns = [
        r"user\s+['\"]([^'\"]+)['\"]",
        r"(?:signed in|logged in)\s+as\s+(?:user\s+)?['\"]?([A-Za-z0-9._-]+)['\"]?",
        r"user\s+([A-Za-z0-9._-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def get_ollama_user() -> str:
    proc = run(["ollama", "signin"], check=False, capture=True)
    return parse_ollama_user((proc.stdout or "").strip())


def write_modelfile() -> None:
    MODELSPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELSPEC_PATH.write_text(
        "\n".join(
            [
                f"FROM {PORTABLE_GGUF}",
                "",
                'TEMPLATE """<|im_start|>user',
                "{{ .Prompt }}<|im_end|>",
                "<|im_start|>assistant",
                '"""',
                "",
                "PARAMETER temperature 0.7",
                "PARAMETER top_k 50",
                "PARAMETER top_p 1",
                "PARAMETER repeat_penalty 1",
                "PARAMETER repeat_last_n 0",
                "PARAMETER num_ctx 128",
                'PARAMETER stop "<|im_end|>"',
                'PARAMETER stop "<|im_start|>"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_gguf(py: Path) -> None:
    info("Preparing canonical tensors")
    run(
        [
            str(py),
            str(REPO_ROOT / "tools" / "prepare_gguf.py"),
            "--checkpoint",
            str(HF_MODEL_DIR / "pytorch_model.bin"),
            "--config",
            str(HF_MODEL_DIR / "config.json"),
            "--out-dir",
            str(GGUF_PREP_DIR),
        ],
        check=True,
    )

    info("Writing GGUF")
    run(
        [
            str(py),
            str(REPO_ROOT / "tools" / "export_gguf.py"),
            "--canonical",
            str(GGUF_PREP_DIR / "canonical_state.pt"),
            "--metadata",
            str(GGUF_PREP_DIR / "metadata.json"),
            "--tokenizer",
            str(HF_MODEL_DIR / "tokenizer.json"),
            "--arch",
            "gpt2",
            "--out",
            str(PORTABLE_GGUF),
        ],
        check=True,
    )


def create_and_test_ollama_model(local_name: str) -> None:
    write_modelfile()

    info(f"Creating local Ollama model: {local_name}")
    run(["ollama", "create", local_name, "-f", str(MODELSPEC_PATH)], check=True)

    info("Running smoke test prompt")
    run(["ollama", "run", local_name, "hi guppy"], check=True)


def push_model(local_name: str, namespace: str, model_name: str) -> str:
    remote_name = f"{namespace}/{model_name}"
    if local_name != remote_name:
        info(f"Tagging {local_name} -> {remote_name}")
        run(["ollama", "cp", local_name, remote_name], check=True)

    info(f"Pushing model: {remote_name}")
    run(["ollama", "push", remote_name], check=True)
    return f"https://ollama.com/{remote_name}"


def print_preflight() -> None:
    print("\nPreflight status")
    print(f"- Platform: {platform.system()} {platform.release()}")
    print(f"- Repo root: {REPO_ROOT}")
    print(f"- Required model files dir: {HF_MODEL_DIR}")


def main() -> None:
    print("GuppyLM -> Ollama publisher")
    print_preflight()

    py = ensure_venv()
    info(f"Using Python: {py}")

    ensure_python_packages(py)
    ensure_model_assets(py)
    ensure_ollama_ready()
    ensure_ollama_login()

    local_name = prompt("Local Ollama model name", "guppylm")
    username = prompt("Ollama username/namespace", get_ollama_user())
    if not username:
        fail("Username is required for push.")
    remote_model = prompt("Remote model name", "guppylm")

    info("Starting build pipeline")
    build_gguf(py)
    create_and_test_ollama_model(local_name)

    url = push_model(local_name, username, remote_model)

    print("\nDone")
    print(f"- Local model: {local_name}")
    print(f"- Remote model: {username}/{remote_model}")
    print(f"- URL: {url}")


if __name__ == "__main__":
    main()
