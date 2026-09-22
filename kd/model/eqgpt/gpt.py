"""
GPT (decoder-only Transformer) architecture for EqGPT.

Adapted from `kd/dataset/EqGPT/code/gpt_model.py` (original EqGPT repo).
Architecture and autoregressive sampling logic (`GPT.step`) are ported as-is;
the only substantive change is that `device` is a module-level variable set
via `set_device()` instead of being hardcoded to `torch.device("cuda")`
(the original repo assumes a CUDA machine everywhere).
"""

import random

import numpy as np
import torch
from torch import nn

from .vocab import vocab_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(new_device) -> None:
    """Override the device used by GPT modules (default: cuda if available, else cpu)."""
    global device
    device = torch.device(new_device)


MAX_POS = 50
D_MODEL = 768
D_FF = 2048
D_K = D_V = 64
N_LAYERS = 6
N_HEADS = 8
CLIP = 1


def get_attn_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor) -> torch.Tensor:
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # PAD token is id 0
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq: torch.Tensor) -> torch.Tensor:
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask.to(device)


class ScaledDotProductAttention(nn.Module):
    def forward(self, q, k, v, attn_mask):
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(D_K)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(D_MODEL, D_K * N_HEADS, bias=False)
        self.W_K = nn.Linear(D_MODEL, D_K * N_HEADS, bias=False)
        self.W_V = nn.Linear(D_MODEL, D_V * N_HEADS, bias=False)
        self.fc = nn.Linear(N_HEADS * D_V, D_MODEL, bias=False)
        self.layernorm = nn.LayerNorm(D_MODEL)

    def forward(self, input_q, input_k, input_v, attn_mask):
        residual, batch_size = input_q, input_q.size(0)
        q = self.W_Q(input_q).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        k = self.W_K(input_k).view(batch_size, -1, N_HEADS, D_K).transpose(1, 2)
        v = self.W_V(input_v).view(batch_size, -1, N_HEADS, D_V).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, N_HEADS, 1, 1)
        context, attn = ScaledDotProductAttention()(q, k, v, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, N_HEADS * D_V)
        output = self.fc(context)
        return self.layernorm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, D_FF, bias=False),
            nn.ReLU(),
            nn.Linear(D_FF, D_MODEL, bias=False),
        )
        self.layernorm = nn.LayerNorm(D_MODEL)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layernorm(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, dec_self_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tgt_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_POS, D_MODEL)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])

    def forward(self, dec_inputs):
        seq_len = dec_inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand_as(dec_inputs)

        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)
        dec_self_attn_mask = torch.gt(dec_self_attn_pad_mask + dec_self_attn_subsequence_mask, 0)

        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


class GPT(nn.Module):
    """Decoder-only Transformer over a small closed vocabulary of PDE-term symbols."""

    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.projection = nn.Linear(D_MODEL, vocab_size)

    def forward(self, dec_inputs):
        dec_outputs, dec_self_attns = self.decoder(dec_inputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), dec_self_attns

    def step(self, sentence, mask_invalid):
        """
        Autoregressively sample the next token given `sentence` (list of
        token ids so far), masking out invalid vocabulary entries and
        enforcing the alternating operator/term-slot grammar (even sequence
        position -> operator, odd -> term).
        """
        dec_input = torch.tensor(sentence, dtype=torch.long, device=device).unsqueeze(0)
        dec_outputs, _ = self.decoder(dec_input)
        projected = self.projection(dec_outputs)

        prob = nn.functional.softmax(projected, dim=2)
        prob = prob[0, -1].squeeze(0)
        prob_filter = prob * mask_invalid

        if dec_input.shape[1] % 2 == 0:
            prob_filter[0] = 0
            prob_filter[5:] = 0
            prob_filter = prob_filter / torch.sum(prob_filter)
        else:
            prob_filter[0:6] = 0
            prob_filter = prob_filter / torch.sum(prob_filter)

        prob_filter = prob_filter.cpu().data.numpy()
        if random.random() <= 0.8:
            next_step = np.random.choice(np.arange(0, vocab_size, 1), p=prob_filter.ravel())
        else:
            valid_step = np.where(prob_filter != 0)[0]
            next_step = np.random.choice(valid_step)
        return next_step, prob
