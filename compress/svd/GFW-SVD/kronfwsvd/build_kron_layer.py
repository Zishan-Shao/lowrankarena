import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from krondecomp_cupy import get_kron_factors


def load_tensor(path, key):
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grads-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    grads_dir = Path(args.grads_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grad_files = sorted(grads_dir.glob("grads_batch_*.safetensors"))
    if not grad_files:
        raise RuntimeError(f"No gradient files found in {grads_dir}")

    with safe_open(str(grad_files[0]), framework="pt", device="cpu") as f:
        layer_names = sorted(f.keys())

    print(f"Gradient files: {len(grad_files)}")
    print(f"Layers: {len(layer_names)}")

    for layer_name in layer_names:
        print(f"\n=== {layer_name} ===")

        grads = []
        for path in grad_files:
            g = load_tensor(path, layer_name)
            print(path.name, tuple(g.shape), g.dtype)
            grads.append(g.float().numpy())

        # maxim convention:
        # first returned factor  -> output x output
        # second returned factor -> input x input
        out_factor, in_factor = get_kron_factors(
            grads,
            top_k=args.top_k,
            layer_name=layer_name,
            device_id=0,
        )

        # LowRankArena convention:
        # XF = input-side
        # YF = output-side
        XF = torch.from_numpy(in_factor.get()).float().contiguous()
        YF = torch.from_numpy(out_factor.get()).float().contiguous()

        print("XF:", tuple(XF.shape))
        print("YF:", tuple(YF.shape))

        filename = layer_name.replace(".", "_") + ".safetensors"

        save_file(
            {
                "XF": XF,
                "YF": YF,
            },
            str(output_dir / filename),
        )

        del grads, XF, YF, out_factor, in_factor

    print("\nDONE:", output_dir)


if __name__ == "__main__":
    main()
