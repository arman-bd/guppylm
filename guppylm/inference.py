"""GuppyLM inference — simple chat."""

import json
import time
import uuid

import torch
from tokenizers import Tokenizer

from .config import GuppyConfig
from .model import GuppyLM


class GuppyInference:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.device = torch.device(device)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        import os
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Load config.json from same directory as the model file
        config_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        config_path = os.path.join(config_dir, "config.json")

        # Extract state_dict — handle both legacy and standard formats
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        # Load config — try config.json first, fall back to embedded config
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            # Support both HF standard keys and our own keys
            self.config = GuppyConfig(
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
            self.config = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid_fields})
        else:
            print("Warning: No config found, using defaults")
            self.config = GuppyConfig()

        self.model = GuppyLM(self.config).to(self.device)
        filtered = {k: v for k, v in state_dict.items() if k in self.model.state_dict()}
        self.model.load_state_dict(filtered)
        self.model.eval()

        total, _ = self.model.param_count()
        print(f"GuppyLM loaded: {total/1e6:.1f}M params")

    def chat_completion(self, messages, temperature=0.7, max_tokens=64,
                        top_k=50, **kwargs):
        """Chat completion — takes messages, returns response."""
        prompt = self._format_prompt(messages)
        input_ids = self.tokenizer.encode(prompt).ids
        prompt_tokens = len(input_ids)
        input_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        output_t, _ = self.model.generate(input_t, max_tokens, temperature, top_k)
        output_text = self.tokenizer.decode(output_t[0].tolist()[prompt_tokens:])
        # Truncate at first <|im_end|> — don't let the model leak into the next turn
        if "<|im_end|>" in output_text:
            output_text = output_text.split("<|im_end|>")[0]
        # Also strip any <|im_start|> fragments
        if "<|im_start|>" in output_text:
            output_text = output_text.split("<|im_start|>")[0]
        resp_text = output_text.strip()

        return {
            "choices": [{
                "message": {"role": "assistant", "content": resp_text},
            }],
        }

    def _format_prompt(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content") or ""
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Chat with Guppy")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--prompt", "-p", help="Single prompt mode: ask one question and exit")
    args = p.parse_args()

    engine = GuppyInference(args.checkpoint, args.tokenizer, args.device)

    if args.prompt:
        result = engine.chat_completion([{"role": "user", "content": args.prompt}])
        print(result["choices"][0]["message"]["content"])
        return

    print("\nGuppy Chat (type 'quit' to exit)")
    while True:
        inp = input("\nYou> ").strip()
        if inp.lower() in ("quit", "exit", "q"):
            break
        result = engine.chat_completion([{"role": "user", "content": inp}])
        msg = result["choices"][0]["message"]
        if msg.get("content"):
            print(f"Guppy> {msg['content']}")


if __name__ == "__main__":
    main()
