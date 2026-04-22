#!/usr/bin/env python3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = REPO_ROOT / "lm-evaluation-harness"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

# Expose the class under __main__ so torch.load can unpickle checkpoints
# saved from hf_prune.py when it was executed as a script.
from hf_prune import LlamaForCausalLMWithGen  # noqa: E402,F401
from main import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    main()
