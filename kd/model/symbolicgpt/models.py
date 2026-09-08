"""GPT model, adapted from archive/kd/model/SymbolicGPT/models.py (original:
https://github.com/mojivalipour/symbolicgpt, minGPT-style architecture).

Changes vs. the original:
- `PointNetConfig`'s constructor parameter was spelled `varibleEmbedding`
  while every caller in the original repo passed the correctly-spelled
  `variableEmbedding` -- because the constructor also accepted **kwargs,
  this silently created a second, unread attribute instead of raising, so
  `variableEmbedding='LEA_EMB'` mode was never actually reachable. Renamed
  to `variable_embedding` everywhere (constructor param, stored attribute,
  and the one place `GPT.forward` reads it) so the value the caller passes
  is the value that's actually used.
- Dropped `PointNet` (an alternate point-cloud encoder defined in the
  original but never instantiated -- `GPT` always builds a `tNet` instead)
  and `GPT1Config` (a 125M-param preset, unused by every caller).
- `embeddingSize`/`numberofPoints`/`numberofVars`/`numberofYs`/`method` on
  `PointNetConfig` and `n_layer`/`n_head`/`n_embd`/`padding_idx` on
  `GPTConfig` keep their original camelCase names since they're set via
  `**kwargs`-driven `setattr`, matching how `kd_symbolicgpt.py` constructs
  them.
"""

import math
import logging

import random
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """Base GPT config, params common to all GPT versions."""
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer with a projection at the end."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """An unassuming Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PointNetConfig:
    """Base PointNet config."""

    def __init__(self, embeddingSize, numberofPoints, numberofVars,
                 numberofYs, method='GPT', variable_embedding='NOT_VAR',
                 **kwargs):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # number of points
        self.numberofVars = numberofVars  # input dimension (Xs)
        self.numberofYs = numberofYs  # output dimension (Ys)
        self.method = method
        self.variable_embedding = variable_embedding

        for k, v in kwargs.items():
            setattr(self, k, v)


class tNet(nn.Module):
    """
    Point-cloud encoder from the original PointNet paper (Qi et al. 2017):
    per-point 1D convolutions + global max pooling + a small MLP head.

    Input:  [batch, numberofVars+numberofYs, numberofPoints]
    Output: [batch, embeddingSize]
    """

    def __init__(self, config):
        super().__init__()

        self.activation_func = F.relu
        self.num_units = config.embeddingSize

        self.conv1 = nn.Conv1d(config.numberofVars + config.numberofYs, self.num_units, 1)
        self.conv2 = nn.Conv1d(self.num_units, 2 * self.num_units, 1)
        self.conv3 = nn.Conv1d(2 * self.num_units, 4 * self.num_units, 1)
        self.fc1 = nn.Linear(4 * self.num_units, 2 * self.num_units)
        self.fc2 = nn.Linear(2 * self.num_units, self.num_units)

        self.input_batch_norm = nn.BatchNorm1d(config.numberofVars + config.numberofYs)

        self.bn1 = nn.BatchNorm1d(self.num_units)
        self.bn2 = nn.BatchNorm1d(2 * self.num_units)
        self.bn3 = nn.BatchNorm1d(4 * self.num_units)
        self.bn4 = nn.BatchNorm1d(2 * self.num_units)
        self.bn5 = nn.BatchNorm1d(self.num_units)

    def forward(self, x):
        x = self.input_batch_norm(x)
        x = self.activation_func(self.bn1(self.conv1(x)))
        x = self.activation_func(self.bn2(self.conv2(x)))
        x = self.activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4 * self.num_units

        x = self.activation_func(self.bn4(self.fc1(x)))
        x = self.activation_func(self.bn5(self.fc2(x)))

        return x


class GPT(nn.Module):
    """The full GPT language model, with a context size of block_size."""

    def __init__(self, config, pointNetConfig=None):
        super().__init__()

        self.config = config
        self.pointNetConfig = pointNetConfig
        self.pointNet = None

        embeddingSize = config.n_embd
        if self.pointNetConfig is not None:

            if self.pointNetConfig.method == 'EMB_CAT':
                embeddingSize = config.n_embd // 2  # if concatenation

            # POINT embedding must have the same size as token/position embedding
            if self.pointNetConfig.embeddingSize != embeddingSize:
                self.pointNetConfig.embeddingSize = embeddingSize

            self.pointNet = tNet(self.pointNetConfig)

            self.vars_emb = nn.Embedding(self.pointNetConfig.numberofVars + 1, embeddingSize)

        if self.pointNetConfig.method == 'EMB_CON':
            self.block_size = config.block_size + 1  # add a first token
            config.block_size += 1
        else:
            self.block_size = config.block_size

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, embeddingSize, padding_idx=self.config.padding_idx)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, embeddingSize))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        if self.pointNetConfig.method == 'OUT_CAT':
            self.head = nn.Linear(config.n_embd * 2, config.vocab_size, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        Separates all parameters into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and
        layernorm/embedding/batchnorm weights). Returns the AdamW optimizer.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, points=None, variables=None, tokenizer=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]

        if points is not None and self.pointNet is not None:
            points_embeddings = self.pointNet(points)

            if variables is not None and self.pointNetConfig.variable_embedding == 'LEA_EMB':
                variables_embeddings = self.vars_emb(variables)
                points_embeddings = points_embeddings + variables_embeddings

            points_embeddings = points_embeddings.unsqueeze(1)

            if self.pointNetConfig.method == 'EMB_CON':
                input_embedding = token_embeddings + position_embeddings
                input_embedding = torch.cat((points_embeddings, input_embedding), dim=1)
            else:
                points_embeddings = torch.tile(points_embeddings, (1, token_embeddings.shape[1], 1))

                if self.pointNetConfig.method == 'EMB_SUM':
                    input_embedding = token_embeddings + position_embeddings + points_embeddings
                elif self.pointNetConfig.method == 'EMB_CAT':
                    input_embedding = token_embeddings + position_embeddings
                    input_embedding = torch.cat((input_embedding, points_embeddings), dim=-1)
                else:
                    input_embedding = token_embeddings + position_embeddings
        else:
            input_embedding = token_embeddings + position_embeddings
            points_embeddings = None

        x = self.drop(input_embedding)
        x = self.blocks(x)
        x = self.ln_f(x)

        if self.pointNetConfig.method == 'OUT_SUM':
            x = x + points_embeddings
        elif self.pointNetConfig.method == 'OUT_CAT':
            x = torch.cat((x, points_embeddings), dim=-1)
        elif self.pointNetConfig.method == 'EMB_CON':
            x = x[:, 1:, :]  # remove the first (point-embedding) token

        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=self.config.padding_idx)

        return logits, loss
