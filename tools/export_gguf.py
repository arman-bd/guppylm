"""
Write a first-pass GGUF file from canonical GuppyLM tensors.
"""

import argparse
import json
import os


def load_deps():
    try:
        import torch
        import numpy as np
        import gguf
        return torch, np, gguf
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install in .venv: pip install torch numpy gguf") from exc


def build_token_list(tokenizer_json_path, vocab_size, gguf_module):
    tok = json.load(open(tokenizer_json_path))
    vocab = tok["model"]["vocab"]
    merges = tok["model"].get("merges", [])
    added = tok.get("added_tokens", [])

    by_id = [""] * vocab_size
    for token, idx in vocab.items():
        if 0 <= idx < vocab_size:
            by_id[idx] = token

    # Default all tokens to NORMAL. Unused rows are overwritten below.
    tok_types = [int(gguf_module.TokenType.NORMAL)] * vocab_size
    added_ids = {}
    for item in added:
        if isinstance(item, dict):
            tid = item.get("id")
            if isinstance(tid, int) and 0 <= tid < vocab_size:
                added_ids[tid] = item

    for i, t in enumerate(by_id):
        if t == "":
            by_id[i] = f"<unused_{i}>"
            tok_types[i] = int(gguf_module.TokenType.UNUSED)
            continue
        info = added_ids.get(i)
        if info and info.get("special", False):
            tok_types[i] = int(gguf_module.TokenType.CONTROL)

    merge_lines = []
    for m in merges:
        if isinstance(m, list) and len(m) == 2:
            merge_lines.append(f"{m[0]} {m[1]}")
        elif isinstance(m, str):
            merge_lines.append(m)
    return by_id, tok_types, merge_lines


def main():
    parser = argparse.ArgumentParser(description="Write first-pass GuppyLM GGUF")
    parser.add_argument("--canonical", default="artifacts/guppylm_gguf_prep/canonical_state.pt")
    parser.add_argument("--metadata", default="artifacts/guppylm_gguf_prep/metadata.json")
    parser.add_argument("--tokenizer", default="data/hf/guppylm-9M/tokenizer.json")
    parser.add_argument("--out", default="artifacts/guppylm_gguf_prep/guppylm-f32.gguf")
    parser.add_argument("--arch", default="guppylm", choices=["guppylm", "gpt2"])
    args = parser.parse_args()

    torch, np, gguf = load_deps()

    if not os.path.exists(args.canonical):
        raise FileNotFoundError(f"Missing canonical tensors: {args.canonical}")
    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f"Missing metadata: {args.metadata}")
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(f"Missing tokenizer: {args.tokenizer}")

    meta = json.load(open(args.metadata))
    state = torch.load(args.canonical, map_location="cpu")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = gguf.GGUFWriter(args.out, arch=args.arch)

    writer.add_name("GuppyLM")
    writer.add_context_length(int(meta["context_length"]))
    writer.add_embedding_length(int(meta["embedding_length"]))
    writer.add_block_count(int(meta["block_count"]))
    writer.add_feed_forward_length(int(meta["feed_forward_length"]))
    writer.add_head_count(int(meta["attention_head_count"]))
    if hasattr(writer, "add_head_count_kv"):
        writer.add_head_count_kv(int(meta["attention_head_count"]))
    writer.add_layer_norm_eps(float(meta["attention_layer_norm_epsilon"]))
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_causal_attention(True)

    writer.add_bos_token_id(int(meta["bos_token_id"]))
    writer.add_eos_token_id(int(meta["eos_token_id"]))
    writer.add_pad_token_id(int(meta["pad_token_id"]))
    writer.add_tokenizer_model("gpt2")

    if hasattr(writer, "add_tokenizer_pre"):
        writer.add_tokenizer_pre("default")

    tokens, tok_types, merges = build_token_list(args.tokenizer, int(meta["vocab_size"]), gguf)
    writer.add_token_list(tokens)
    if hasattr(writer, "add_token_types"):
        writer.add_token_types(tok_types)
    if merges:
        writer.add_token_merges(merges)

    for name in sorted(state.keys()):
        arr = state[name].detach().cpu().numpy().astype(np.float32, copy=False)
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    print(f"Wrote GGUF: {args.out}")
    print(f"Tensors: {len(state)}")


if __name__ == "__main__":
    main()
