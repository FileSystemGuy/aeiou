"""ABSTRACTS.md §11 row 4b: a random-weight GPT-2-shaped model saved as safetensors shards (no download)."""
import sys, torch
from transformers import GPT2Config, GPT2LMHeadModel
torch.manual_seed(1)
cfg = GPT2Config(n_layer=6, n_embd=1024, n_head=16, vocab_size=32000, n_positions=1024)
GPT2LMHeadModel(cfg).save_pretrained(sys.argv[1], max_shard_size=sys.argv[2] if len(sys.argv) > 2 else "100MB")
