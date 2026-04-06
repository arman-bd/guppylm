"""Generate the GuppyLM Colab training notebook."""

import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_file(path):
    with open(path) as f:
        return f.read()


def read_for_colab(path):
    """Read a Python file and flatten relative imports for Colab."""
    content = read_file(path)
    content = re.sub(r'from \.(\w+) import', r'from \1 import', content)
    return content


def cell(source, cell_type="code"):
    lines = source.split("\n")
    formatted = [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]
    base = {"cell_type": cell_type, "metadata": {}, "source": formatted}
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    return base


def md(text):
    return cell(text, "markdown")


def code(text):
    return cell(text, "code")


# Source files to embed in the notebook
FILES = [
    ("config.py",    "guppylm/config.py"),
    ("model.py",     "guppylm/model.py"),
    ("dataset.py",   "guppylm/dataset.py"),
    ("train.py",     "guppylm/train.py"),
    ("inference.py", "guppylm/inference.py"),
]


def build():
    cells = []

    # ══════════════════════════════════════════════════════════════════
    #  HEADER
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "# GuppyLM — Your Friendly Fish\n"
        "\n"
        "Train a ~9M parameter LLM that talks like a small fish.\n"
        "\n"
        "**What this notebook does:**\n"
        "1. Downloads 60K fish conversation dataset from HuggingFace\n"
        "2. Trains a BPE tokenizer on the data\n"
        "3. Trains a 6-layer vanilla transformer (8.7M params)\n"
        "4. Tests the model with sample conversations\n"
        "\n"
        "**Architecture:** 6 layers, 384 dim, 6 heads, ReLU FFN, LayerNorm, 4096 vocab\n"
        "\n"
        "**Runtime:** ~5 min on T4 GPU\n"
        "\n"
        "**Result:** A fish that speaks in short lowercase sentences about water, food, and light."
    ))

    # ══════════════════════════════════════════════════════════════════
    #  1. SETUP
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 1. Setup\n"
        "\n"
        "Install dependencies and create a clean working directory."
    ))

    cells.append(code(
        "!pip install -q torch tokenizers tqdm numpy datasets huggingface_hub\n"
        "\n"
        "import torch\n"
        "print(f'PyTorch {torch.__version__}')\n"
        "print(f'CUDA: {torch.cuda.is_available()}')\n"
        "if torch.cuda.is_available():\n"
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
    ))

    cells.append(code(
        "import os, shutil\n"
        "\n"
        "# Start fresh — removes stale files from previous runs\n"
        "if os.path.exists('/content/guppy'):\n"
        "    shutil.rmtree('/content/guppy')\n"
        "os.makedirs('/content/guppy')\n"
        "os.chdir('/content/guppy')\n"
        "print(f'Working dir: {os.getcwd()}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  2. SOURCE FILES
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 2. Source Files\n"
        "\n"
        "Write the model code to disk. These are the only files needed:\n"
        "- `config.py` — model and training hyperparameters\n"
        "- `model.py` — transformer architecture\n"
        "- `dataset.py` — data loading and batching\n"
        "- `train.py` — training loop\n"
        "- `inference.py` — chat interface"
    ))

    for display_name, src_path in FILES:
        full_path = os.path.join(PROJECT_ROOT, src_path)
        content = read_for_colab(full_path)
        cells.append(code(f"%%writefile {display_name}\n{content}"))

    # ══════════════════════════════════════════════════════════════════
    #  3. PREPARE DATA
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 3. Prepare Data\n"
        "\n"
        "Download the fish conversation dataset from HuggingFace and train a BPE tokenizer.\n"
        "\n"
        "The dataset has 60K single-turn conversations across 60 topics:\n"
        "greetings, food, temperature, water, tank life, emotions, philosophy (fish-level), and more.\n"
        "\n"
        "Each sample is formatted as ChatML:\n"
        "```\n"
        "<|im_start|>user\n"
        "hi guppy<|im_end|>\n"
        "<|im_start|>assistant\n"
        "hello. the water is nice today.<|im_end|>\n"
        "```"
    ))

    cells.append(code(
        "import json, os\n"
        "from datasets import load_dataset\n"
        "from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors\n"
        "\n"
        "# ── Download from HuggingFace ──\n"
        "HF_DATASET = 'arman-bd/guppylm-60k-generic'\n"
        "ds = load_dataset(HF_DATASET)\n"
        "print(f'Downloaded: {len(ds[\"train\"]):,} train, {len(ds[\"test\"]):,} test samples')\n"
        "\n"
        "# ── Format into ChatML and save as JSONL ──\n"
        "os.makedirs('data', exist_ok=True)\n"
        "texts = []\n"
        "\n"
        "for split, path in [('train', 'data/train.jsonl'), ('test', 'data/eval.jsonl')]:\n"
        "    with open(path, 'w') as f:\n"
        "        for row in ds[split]:\n"
        "            text = (\n"
        "                f'<|im_start|>user\\n{row[\"input\"]}<|im_end|>\\n'\n"
        "                f'<|im_start|>assistant\\n{row[\"output\"]}<|im_end|>'\n"
        "            )\n"
        "            f.write(json.dumps({'text': text, 'category': row['category']}) + '\\n')\n"
        "            texts.append(text)\n"
        "    print(f'  {path}: {len(ds[split]):,} samples')\n"
        "\n"
        "# ── Train BPE tokenizer on the data ──\n"
        "tokenizer = Tokenizer(models.BPE())\n"
        "tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)\n"
        "tokenizer.decoder = decoders.ByteLevel()\n"
        "\n"
        "trainer = trainers.BpeTrainer(\n"
        "    vocab_size=4096,\n"
        "    special_tokens=['<pad>', '<|im_start|>', '<|im_end|>'],\n"
        "    min_frequency=2,\n"
        "    show_progress=True,\n"
        ")\n"
        "tokenizer.train_from_iterator(texts, trainer)\n"
        "tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)\n"
        "tokenizer.save('data/tokenizer.json')\n"
        "print(f'  Tokenizer: {tokenizer.get_vocab_size()} tokens')\n"
        "\n"
        "# ── Preview ──\n"
        "with open('data/train.jsonl') as f:\n"
        "    sample = json.loads(f.readline())\n"
        "print(f'\\nSample ({sample[\"category\"]}):\\n{sample[\"text\"]}')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  4. VERIFY ARCHITECTURE
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 4. Verify Architecture\n"
        "\n"
        "Quick sanity check — build the model, print param count, run a dummy forward pass."
    ))

    cells.append(code(
        "from config import GuppyConfig\n"
        "from model import GuppyLM\n"
        "import torch\n"
        "\n"
        "config = GuppyConfig()\n"
        "model = GuppyLM(config)\n"
        "print(model.param_summary())\n"
        "print(f'  Layers: {config.n_layers}, Heads: {config.n_heads}, FFN: {config.ffn_hidden}')\n"
        "print(f'  Vocab: {config.vocab_size}, Max seq: {config.max_seq_len}')\n"
        "\n"
        "# Dummy forward pass\n"
        "x = torch.randint(0, config.vocab_size, (2, 32))\n"
        "logits, _ = model(x)\n"
        "print(f'  Forward pass: {x.shape} -> {logits.shape} OK')\n"
        "del model"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  5. TRAIN
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 5. Train\n"
        "\n"
        "10,000 steps with cosine LR schedule. Takes ~2 min on T4.\n"
        "\n"
        "The model learns to:\n"
        "- Respond in short, lowercase sentences\n"
        "- Stay in character as a fish\n"
        "- Cover 60 different conversation topics\n"
        "- Stop generating at the right time (learn the `<|im_end|>` token)"
    ))

    cells.append(code("from train import train\ntrain()"))

    # ══════════════════════════════════════════════════════════════════
    #  6. TEST
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 6. Test\n"
        "\n"
        "Chat with the trained model. Each message is independent (single-turn)."
    ))

    cells.append(code(
        "from inference import GuppyInference\n"
        "import torch\n"
        "\n"
        "engine = GuppyInference(\n"
        "    'checkpoints/best_model.pt', 'data/tokenizer.json',\n"
        "    device='cuda' if torch.cuda.is_available() else 'cpu'\n"
        ")\n"
        "\n"
        "def chat(prompt):\n"
        "    r = engine.chat_completion([{'role': 'user', 'content': prompt}], max_tokens=64)\n"
        "    return r['choices'][0]['message'].get('content', '').strip()\n"
        "\n"
        "# Test across different topics\n"
        "tests = [\n"
        "    ('hi guppy',                      'greeting'),\n"
        "    ('are you hungry',                'food'),\n"
        "    ('it is really hot today',        'temperature'),\n"
        "    ('how is the water',              'water'),\n"
        "    ('do you like bubbles',           'bubbles'),\n"
        "    ('what is the internet',          'confused'),\n"
        "    ('do you get lonely',             'lonely'),\n"
        "    ('the cat is looking at you',     'cat'),\n"
        "    ('tell me a joke',                'joke'),\n"
        "    ('what do you dream about',       'dreams'),\n"
        "    ('do you love me',                'love'),\n"
        "    ('what is the meaning of life',   'meaning'),\n"
        "    ('sorry i tapped the glass',      'glass_tap'),\n"
        "    ('it is raining outside',         'rain'),\n"
        "    ('goodnight guppy',               'night'),\n"
        "]\n"
        "\n"
        "print(f'{\"Topic\":<12s}  {\"You\":<35s}  Guppy')\n"
        "print('=' * 100)\n"
        "for prompt, topic in tests:\n"
        "    reply = chat(prompt)\n"
        "    print(f'{topic:<12s}  {prompt:<35s}  {reply[:128]}')\n"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  7. EXPORT & UPLOAD TO HUGGINGFACE
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 7. Export & Upload to HuggingFace\n"
        "\n"
        "Export the model in both PyTorch and ONNX (quantized uint8, ~9 MB) formats,\n"
        "then upload everything to HuggingFace in one go.\n"
        "\n"
        "Set your token and repo below."
    ))

    cells.append(code(
        "!pip install -q onnx onnxruntime onnxscript\n"
        "\n"
        "from huggingface_hub import HfApi, login\n"
        "import torch, json, os, shutil\n"
        "from config import GuppyConfig\n"
        "from model import GuppyLM\n"
        "\n"
        "HF_TOKEN = os.environ.get('HF_TOKEN', '')  # Or paste your token here\n"
        "HF_REPO = os.environ.get('HF_REPO', 'arman-bd/guppylm-9M')  # Or change this\n"
        "\n"
        "# Load checkpoint\n"
        "ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)\n"
        "cfg = ckpt['config']\n"
        "os.makedirs('hf_export', exist_ok=True)\n"
        "\n"
        "# ── PyTorch format ──\n"
        "torch.save(ckpt['model_state_dict'], 'hf_export/pytorch_model.bin')\n"
        "\n"
        "with open('hf_export/config.json', 'w') as f:\n"
        "    json.dump({\n"
        "        'model_type': 'guppylm',\n"
        "        'architectures': ['GuppyLM'],\n"
        "        'vocab_size': cfg['vocab_size'],\n"
        "        'max_position_embeddings': cfg['max_seq_len'],\n"
        "        'hidden_size': cfg['d_model'],\n"
        "        'num_hidden_layers': cfg['n_layers'],\n"
        "        'num_attention_heads': cfg['n_heads'],\n"
        "        'intermediate_size': cfg['ffn_hidden'],\n"
        "        'hidden_dropout_prob': cfg.get('dropout', 0.1),\n"
        "        'pad_token_id': cfg['pad_id'],\n"
        "        'bos_token_id': cfg['bos_id'],\n"
        "        'eos_token_id': cfg['eos_id'],\n"
        "    }, f, indent=2)\n"
        "\n"
        "shutil.copy('data/tokenizer.json', 'hf_export/tokenizer.json')\n"
        "print(f'pytorch_model.bin: {os.path.getsize(\"hf_export/pytorch_model.bin\")/1e6:.1f} MB')\n"
        "\n"
        "# ── ONNX format (quantized uint8) ──\n"
        "valid_fields = {f.name for f in GuppyConfig.__dataclass_fields__.values()}\n"
        "config = GuppyConfig(**{k: v for k, v in cfg.items() if k in valid_fields})\n"
        "model = GuppyLM(config)\n"
        "model.load_state_dict(ckpt['model_state_dict'])\n"
        "model.eval()\n"
        "\n"
        "dummy = torch.randint(0, config.vocab_size, (1, 32))\n"
        "fp32_path = 'hf_export/model_fp32.onnx'\n"
        "torch.onnx.export(\n"
        "    model, (dummy,), fp32_path,\n"
        "    input_names=['input_ids'], output_names=['logits'],\n"
        "    dynamic_axes={'input_ids': {0: 'batch', 1: 'seq_len'},\n"
        "                  'logits': {0: 'batch', 1: 'seq_len'}},\n"
        "    opset_version=17,\n"
        ")\n"
        "\n"
        "from onnxruntime.quantization import quantize_dynamic, QuantType\n"
        "quantize_dynamic(fp32_path, 'hf_export/model.onnx', weight_type=QuantType.QUInt8)\n"
        "os.remove(fp32_path)\n"
        "print(f'model.onnx:       {os.path.getsize(\"hf_export/model.onnx\")/1e6:.1f} MB (uint8)')\n"
        "\n"
        "# ── Upload ──\n"
        "if HF_TOKEN:\n"
        "    login(token=HF_TOKEN)\n"
        "    api = HfApi()\n"
        "    api.create_repo(HF_REPO, exist_ok=True)\n"
        "    api.upload_folder(folder_path='hf_export', repo_id=HF_REPO, repo_type='model')\n"
        "    print(f'Done! https://huggingface.co/{HF_REPO}')\n"
        "else:\n"
        "    print('No HF_TOKEN — exported locally to hf_export/')"
    ))

    # ══════════════════════════════════════════════════════════════════
    #  8. DOWNLOAD
    # ══════════════════════════════════════════════════════════════════

    cells.append(md(
        "## 8. Download\n"
        "\n"
        "Or download the model locally as a tar.gz."
    ))

    cells.append(code(
        "import os\n"
        "\n"
        "!cd /content && tar czf guppylm.tar.gz \\\n"
        "    guppy/checkpoints/best_model.pt \\\n"
        "    guppy/checkpoints/config.json \\\n"
        "    guppy/data/tokenizer.json \\\n"
        "    guppy/model.py \\\n"
        "    guppy/config.py \\\n"
        "    guppy/inference.py \\\n"
        "    guppy/hf_export/model.onnx\n"
        "\n"
        "sz = os.path.getsize('/content/guppylm.tar.gz') / 1e6\n"
        "print(f'Package: /content/guppylm.tar.gz ({sz:.1f} MB)')\n"
        "\n"
        "try:\n"
        "    from google.colab import files\n"
        "    files.download('/content/guppylm.tar.gz')\n"
        "except ImportError:\n"
        "    print('Not in Colab — download manually from the file browser.')"
    ))

    # ══════════════════════════════════════════════════════════════════

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "name": "GuppyLM — Train"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


def build_use():
    """Build the use_guppylm notebook — download model from HF and chat."""
    cells = []

    cells.append(md(
        "# GuppyLM — Chat with a Fish\n"
        "\n"
        "Download a pre-trained 9M parameter fish LLM and chat with it. Just run all cells.\n"
        "\n"
        "**Model:** [arman-bd/guppylm-9M](https://huggingface.co/arman-bd/guppylm-9M)"
    ))

    cells.append(code(
        "# Setup + Download\n"
        "!pip install -q torch tokenizers huggingface_hub\n"
        "import os, shutil\n"
        "if os.path.exists('/content/guppy'): shutil.rmtree('/content/guppy')\n"
        "os.makedirs('/content/guppy'); os.chdir('/content/guppy')\n"
        "\n"
        "from huggingface_hub import snapshot_download\n"
        "snapshot_download(repo_id='arman-bd/guppylm-9M', local_dir='.')\n"
        "print('Model downloaded.')"
    ))

    cells.append(code(
        "# Load model\n"
        "from inference import GuppyInference\n"
        "import torch\n"
        "\n"
        "engine = GuppyInference('pytorch_model.bin', 'tokenizer.json',\n"
        "                        device='cuda' if torch.cuda.is_available() else 'cpu')\n"
        "\n"
        "def chat(prompt):\n"
        "    return engine.chat_completion(\n"
        "        [{'role': 'user', 'content': prompt}], max_tokens=64\n"
        "    )['choices'][0]['message'].get('content', '').strip()\n"
        "\n"
        "# Quick test\n"
        "for p in ['hi guppy', 'are you hungry', 'tell me a joke', 'what is the internet', 'goodnight guppy']:\n"
        "    print(f'You> {p}\\nGuppy> {chat(p)}\\n')"
    ))

    cells.append(code(
        "# Interactive chat — type your messages\n"
        "while True:\n"
        "    try:\n"
        "        p = input('You> ').strip()\n"
        "    except (KeyboardInterrupt, EOFError):\n"
        "        break\n"
        "    if not p or p.lower() in ('quit', 'exit', 'q'):\n"
        "        print('Guppy> bye. i will continue being a fish.'); break\n"
        "    print(f'Guppy> {chat(p)}\\n')"
    ))

    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "name": "GuppyLM — Chat"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def write_notebook(nb, filename):
    out = os.path.join(PROJECT_ROOT, filename)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    n = len(nb["cells"])
    sz = os.path.getsize(out) / 1024
    print(f"Generated {out}: {n} cells, {sz:.1f} KB")


if __name__ == "__main__":
    write_notebook(build(), "train_guppylm.ipynb")
    write_notebook(build_use(), "use_guppylm.ipynb")
