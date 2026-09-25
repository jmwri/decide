"""Generates internal/nn/testdata/tiny_ref.safetensors.

A tiny random ModernBERT (HuggingFace transformers) plus the option-scoring
head, run on two packed sequences (one with order-invariant "independent
options" attention, one with plain attention). The file stores the weights,
the option logits, the loss and every parameter's gradient as computed by
PyTorch autograd. The Go tests load it and check the hand-written forward and
backward passes against it.

    pip install torch transformers safetensors
    python gen_ref.py
"""
import json
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers import ModernBertConfig, ModernBertModel

torch.manual_seed(0)
MASK = 99
HALF = 4  # local_attention = 8

cfg = ModernBertConfig(
    vocab_size=100, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
    num_attention_heads=4, global_attn_every_n_layers=3, local_attention=2 * HALF,
    max_position_embeddings=256, pad_token_id=0, attn_implementation="sdpa",
)
enc = ModernBertModel(cfg).double()
for n, p in enc.named_parameters():
    with torch.no_grad():
        if n.endswith("norm.weight"):
            p.copy_(1 + 0.2 * torch.randn_like(p))
        else:
            p.copy_(0.15 * torch.randn_like(p))


class Scorer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.input_norm = nn.LayerNorm(h)
        self.dense = nn.Linear(h, h // 2)
        self.norm = nn.LayerNorm(h // 2)
        self.out_proj = nn.Linear(h // 2, 1)

    def forward(self, x):
        h = self.dense(self.input_norm(x))
        h = self.norm(nn.functional.gelu(h))
        return self.out_proj(h).squeeze(-1)


scorer = Scorer(cfg.hidden_size).double()
for n, p in scorer.named_parameters():
    with torch.no_grad():
        p.copy_((1 + 0.2 * torch.randn_like(p)) if n.endswith("norm.weight") else 0.3 * torch.randn_like(p))


def make_seq(prefix_len, opt_lens, independent):
    ids = [1] + list(torch.randint(2, 90, (prefix_len - 1,)).tolist()) + [98]  # [CLS] ... [SEP]
    mask_pos = []
    for ln in opt_lens:
        mask_pos.append(len(ids))
        ids += [MASK] + list(torch.randint(2, 90, (ln,)).tolist())
    ids.append(98)  # trailing [SEP]
    n = len(ids)
    pos = list(range(n))
    allowed = torch.ones(n, n, dtype=torch.bool)
    if independent:
        last = n - 1
        opt = [-1] * n
        for k, s in enumerate(mask_pos):
            e = mask_pos[k + 1] if k + 1 < len(mask_pos) else last
            for i in range(s, e):
                opt[i] = k
                pos[i] = mask_pos[0] + (i - s)
        for i in range(n):
            for j in range(n):
                allowed[i, j] = (opt[i] == -1 and opt[j] == -1) or (opt[i] != -1 and (opt[j] == -1 or opt[i] == opt[j]))
    return ids, mask_pos, pos, allowed


seqs = [make_seq(9, [5, 7, 4], True), make_seq(6, [6, 8], False)]
targets = [1, 0]

total_loss = 0.0
all_logits = []
debug = {}
for (ids, mask_pos, pos, allowed), tgt in zip(seqs, targets):
    x = torch.tensor([ids])
    p = torch.tensor([pos])
    n = len(ids)
    dist = (p[0][:, None] - p[0][None, :]).abs() <= HALF
    slide = allowed & dist
    hooks = []
    if not all_logits:
        L1 = enc.layers[1]
        hooks.append(L1.attn_norm.register_forward_hook(lambda m_, i_, o_: debug.__setitem__("l1.n1", o_[0].detach().float().contiguous())))
        hooks.append(L1.attn.register_forward_hook(lambda m_, i_, o_: debug.__setitem__("l1.attn", o_[0][0].detach().float().contiguous())))
    out = enc(input_ids=x, position_ids=p, output_hidden_states=True,
              attention_mask={"full_attention": allowed[None, None], "sliding_attention": slide[None, None]})
    for hk in hooks:
        hk.remove()
    if not all_logits:  # first sequence: keep intermediate activations for debugging
        for li, hsd in enumerate(out.hidden_states[:4]):
            debug["hidden.%d" % li] = hsd[0].detach().float().contiguous()
    rows = out.last_hidden_state[0, mask_pos]
    logits = scorer(rows)
    all_logits.append(logits)
    total_loss = total_loss + nn.functional.cross_entropy(logits[None], torch.tensor([tgt]))
loss = total_loss / len(seqs)
loss.backward()

tensors = {}
for n, p in enc.named_parameters():
    tensors["w.model." + n] = p.detach().float().contiguous()
    tensors["g.model." + n] = p.grad.float().contiguous()
for n, p in scorer.named_parameters():
    tensors["w.scorer." + n] = p.detach().float().contiguous()
    tensors["g.scorer." + n] = p.grad.float().contiguous()
tensors["logits"] = torch.cat(all_logits).detach().float()
tensors["loss"] = loss.detach().float().reshape(1)
tensors.update(debug)

meta = {
    "config": {"hidden": 32, "intermediate": 48, "heads": 4, "layers": 4, "vocab": 100, "half_window": HALF,
               "global_theta": 160000.0, "local_theta": 10000.0},
    "mask_id": MASK,
    "seqs": [{"ids": s[0], "independent": ind, "target": t} for s, ind, t in zip(seqs, [True, False], targets)],
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tiny_ref.safetensors")
save_file(tensors, out, metadata={"meta": json.dumps(meta)})
print("wrote", out, "loss", float(loss))
