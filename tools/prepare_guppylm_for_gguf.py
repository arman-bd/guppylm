"""
Prepare GuppyLM checkpoint tensors for a GGUF converter.

This script does not write GGUF directly. It creates a canonical tensor pack
with split Q/K/V weights and metadata so llama.cpp-side conversion is simpler.
"""

import argparse
import json
import os
from typing import Dict


def load_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise SystemExit("Missing dependency: torch") from exc


def canonicalize(state: Dict[str, "torch.Tensor"], n_layers: int):
    out = {}

    out["token_embd.weight"] = state["tok_emb.weight"]
    out["position_embd.weight"] = state["pos_emb.weight"]
    out["output_norm.weight"] = state["norm.weight"]
    out["output_norm.bias"] = state["norm.bias"]
    out["output.weight"] = state["lm_head.weight"]

    for i in range(n_layers):
        p = f"blocks.{i}"
        c = f"blk.{i}"

        out[f"{c}.attn_norm.weight"] = state[f"{p}.norm1.weight"]
        out[f"{c}.attn_norm.bias"] = state[f"{p}.norm1.bias"]
        out[f"{c}.attn_qkv.weight"] = state[f"{p}.attn.qkv.weight"]
        out[f"{c}.attn_qkv.bias"] = state[f"{p}.attn.qkv.bias"]
        out[f"{c}.attn_output.weight"] = state[f"{p}.attn.out.weight"]
        out[f"{c}.attn_output.bias"] = state[f"{p}.attn.out.bias"]
        out[f"{c}.ffn_norm.weight"] = state[f"{p}.norm2.weight"]
        out[f"{c}.ffn_norm.bias"] = state[f"{p}.norm2.bias"]
        out[f"{c}.ffn_up.weight"] = state[f"{p}.ffn.up.weight"]
        out[f"{c}.ffn_up.bias"] = state[f"{p}.ffn.up.bias"]
        out[f"{c}.ffn_down.weight"] = state[f"{p}.ffn.down.weight"]
        out[f"{c}.ffn_down.bias"] = state[f"{p}.ffn.down.bias"]

    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare GuppyLM tensors for GGUF conversion")
    parser.add_argument("--checkpoint", default="data/hf/guppylm-9M/pytorch_model.bin")
    parser.add_argument("--config", default="data/hf/guppylm-9M/config.json")
    parser.add_argument("--out-dir", default="artifacts/guppylm_gguf_prep")
    args = parser.parse_args()

    torch = load_torch()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(args.config) as f:
        cfg = json.load(f)

    n_layers = int(cfg["num_hidden_layers"])
    d_model = int(cfg["hidden_size"])

    os.makedirs(args.out_dir, exist_ok=True)

    state = torch.load(args.checkpoint, map_location="cpu")
    canonical = canonicalize(state, n_layers=n_layers)

    tensor_path = os.path.join(args.out_dir, "canonical_state.pt")
    torch.save(canonical, tensor_path)

    meta = {
        "architecture": "guppylm",
        "context_length": int(cfg["max_position_embeddings"]),
        "embedding_length": d_model,
        "block_count": n_layers,
        "feed_forward_length": int(cfg["intermediate_size"]),
        "attention_head_count": int(cfg["num_attention_heads"]),
        "attention_layer_norm_epsilon": 1.0e-5,
        "vocab_size": int(cfg["vocab_size"]),
        "bos_token_id": int(cfg["bos_token_id"]),
        "eos_token_id": int(cfg["eos_token_id"]),
        "pad_token_id": int(cfg["pad_token_id"]),
        "source_checkpoint": args.checkpoint,
    }
    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    index_path = os.path.join(args.out_dir, "tensor_index.txt")
    with open(index_path, "w") as f:
        for k in sorted(canonical):
            f.write(f"{k}\t{tuple(canonical[k].shape)}\n")

    print(f"Wrote {tensor_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {index_path}")
    print(f"Total canonical tensors: {len(canonical)}")


if __name__ == "__main__":
    main()
