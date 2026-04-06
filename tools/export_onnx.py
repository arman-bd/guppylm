"""
Export GuppyLM to ONNX format for browser inference via onnxruntime-web.

Exports to web/model.onnx (quantized uint8 by default) + web/tokenizer.json
so the browser demo (web/index.html) can load them — works on GitHub Pages.

Quantization shrinks the model from ~35 MB (float32) to ~9 MB (uint8) with
negligible quality loss at this model size.

Usage:
    python tools/export_onnx.py                        # quantized (default)
    python tools/export_onnx.py --no-quantize          # keep float32
    python tools/export_onnx.py --push                 # also upload to HF repo
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "guppylm"))


def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def export_onnx(checkpoint_path, tokenizer_path, output_path, quantize=True, push=False):
    import shutil

    import torch
    from config import GuppyConfig
    from model import GuppyLM

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Resolve config
    config_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    config_json = os.path.join(config_dir, "config.json")

    if os.path.exists(config_json):
        with open(config_json) as f:
            cfg = json.load(f)
        config = GuppyConfig(
            vocab_size=cfg.get("vocab_size", 4096),
            max_seq_len=cfg.get("max_position_embeddings", cfg.get("max_seq_len", 128)),
            d_model=cfg.get("hidden_size", cfg.get("d_model", 384)),
            n_layers=cfg.get("num_hidden_layers", cfg.get("n_layers", 6)),
            n_heads=cfg.get("num_attention_heads", cfg.get("n_heads", 6)),
            ffn_hidden=cfg.get("intermediate_size", cfg.get("ffn_hidden", 768)),
            dropout=cfg.get("hidden_dropout_prob", cfg.get("dropout", 0.1)),
            pad_id=cfg.get("pad_token_id", cfg.get("pad_id", 0)),
            bos_id=cfg.get("bos_token_id", cfg.get("bos_id", 1)),
            eos_id=cfg.get("eos_token_id", cfg.get("eos_id", 2)),
        )
    elif isinstance(ckpt, dict) and "config" in ckpt:
        valid_fields = {f.name for f in GuppyConfig.__dataclass_fields__.values()}
        config = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
    else:
        config = GuppyConfig()

    # Load model
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = GuppyLM(config)
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model.state_dict()})
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"GuppyLM: {total:,} params ({total/1e6:.1f}M)")

    # Export to ONNX — only the forward pass (logits), no loss
    dummy_input = torch.randint(0, config.vocab_size, (1, 32))

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=17,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported {output_path} ({size_mb:.1f} MB, float32)")

    # Quantize to uint8 — ~4x smaller, negligible quality loss at 9M params
    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        fp32_path = output_path.replace(".onnx", "_fp32.onnx")
        os.rename(output_path, fp32_path)

        quantize_dynamic(
            fp32_path,
            output_path,
            weight_type=QuantType.QUInt8,
        )

        q_size_mb = os.path.getsize(output_path) / 1e6
        print(f"Quantized {output_path} ({q_size_mb:.1f} MB, uint8)")
        os.remove(fp32_path)

    # Copy tokenizer alongside the ONNX model (needed for web/ GitHub Pages demo)
    tok_dest = os.path.join(out_dir, "tokenizer.json")
    if os.path.abspath(tokenizer_path) != os.path.abspath(tok_dest):
        shutil.copy2(tokenizer_path, tok_dest)
        print(f"Copied tokenizer to {tok_dest}")

    # Push to HuggingFace
    if push:
        load_env()
        token = os.environ.get("HF_TOKEN")
        repo = os.environ.get("HF_REPO")
        if not token or not repo:
            print("Set HF_TOKEN and HF_REPO in .env to push. Skipping.")
            return

        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=output_path,
            path_in_repo="model.onnx",
            repo_id=repo,
            repo_type="model",
        )
        print(f"Uploaded model.onnx to https://huggingface.co/{repo}")


def main():
    parser = argparse.ArgumentParser(description="Export GuppyLM to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--output", default="web/model.onnx")
    parser.add_argument("--no-quantize", action="store_true", help="Skip uint8 quantization")
    parser.add_argument("--push", action="store_true", help="Upload to HuggingFace repo")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.tokenizer, args.output,
                quantize=not args.no_quantize, push=args.push)


if __name__ == "__main__":
    main()
