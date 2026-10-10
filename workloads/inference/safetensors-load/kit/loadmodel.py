"""ABSTRACTS.md §11 row 4b: AutoModelForCausalLM.from_pretrained on a local directory of safetensors
shards. The weights stay views of the mapping; `--touch` then copies every parameter, in module order,
as a move to a device would (a stand-in: no GPU here)."""
import sys
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained(sys.argv[-1])
if sys.argv[1] == "--touch":
    print(sum(p.detach().clone().numel() for p in m.parameters()))
