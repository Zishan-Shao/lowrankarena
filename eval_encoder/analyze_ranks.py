import json
import numpy as np

# Budget 0.5 (current)
with open('ars_out/ranks.json') as f:
    ranks_0_5 = json.load(f)

# Get QKV ranks for layer 0 as example
def get_layer_qkv(ranks_dict, layer_idx):
    q = ranks_dict.get(f'bert.encoder.layer.{layer_idx}.attention.self.query', 0)
    k = ranks_dict.get(f'bert.encoder.layer.{layer_idx}.attention.self.key', 0)
    v = ranks_dict.get(f'bert.encoder.layer.{layer_idx}.attention.self.value', 0)
    return q, k, v

print("=== Rank Distribution Analysis ===\n")

# Budget 0.5
ranks_list = [r for r in ranks_0_5.values() if r > 0]
q0, k0, v0 = get_layer_qkv(ranks_0_5, 0)
print(f"Budget 0.5 (current ars_out):")
print(f"  Median={np.median(ranks_list):.0f}, Min={min(ranks_list)}, Max={max(ranks_list)}")
print(f"  Layer0: Q={q0}, K={k0}, V={v0}, MaxDiff={max(q0,k0,v0)-min(q0,k0,v0)}")
print(f"  Range: {max(ranks_list) - min(ranks_list)}")
print()

# Check archived outputs
import os
archived_dir = 'eval_results/archived_adasvd_outputs'
if os.path.exists(archived_dir):
    for budget in ['0.1', '0.3', '0.4']:
        for backend in ['naive', 'flashsvd']:
            ars_dir = f'{archived_dir}/ars_out_FIXED_b{budget}_{backend}'
            if os.path.exists(f'{ars_dir}/ranks.json'):
                with open(f'{ars_dir}/ranks.json') as f:
                    ranks = json.load(f)
                ranks_list = [r for r in ranks.values() if r > 0]
                q0, k0, v0 = get_layer_qkv(ranks, 0)
                print(f"Budget {budget} ({backend}):")
                print(f"  Median={np.median(ranks_list):.0f}, Min={min(ranks_list)}, Max={max(ranks_list)}")
                print(f"  Layer0: Q={q0}, K={k0}, V={v0}, MaxDiff={max(q0,k0,v0)-min(q0,k0,v0)}")
                print(f"  Range: {max(ranks_list) - min(ranks_list)}")
                print()
