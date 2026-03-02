import torch
from eval_encoder.load_compressed_model import load_compressed_model                                                                                                                                 
m, _, info = load_compressed_model('eval_encoder/models/sst2/svd_ra48_rf256_rw256_per_head_naive')                                                                                                   
layer = m.bert.encoder.layer[0]
blk = layer.block
print('Pq.shape:', blk.Pq.shape)
print('R_attn:', blk.Pq.shape[-1])