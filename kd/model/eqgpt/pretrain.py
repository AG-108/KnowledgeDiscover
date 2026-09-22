"""
Supervised GPT pretraining on the PDE-handbook corpus.

Adapted from `kd/dataset/EqGPT/code/train_gpt.py` (original EqGPT repo).
"""

from typing import List

import torch
import torch.utils.data as data
from torch import nn, optim

from .vocab import word2id
from .gpt import GPT, CLIP


class SentenceDataset(data.Dataset):
    """Token-id sequences, split into (decoder_input, decoder_output) shifted pairs."""

    def __init__(self, sequences: List[List[int]]):
        self.sequences = sequences

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {"decoder_input": seq[:-1], "decoder_output": seq[1:]}

    def __len__(self):
        return len(self.sequences)

    def padding_batch(self, batch):
        input_maxlen = max(len(d["decoder_input"]) for d in batch)
        output_maxlen = max(len(d["decoder_output"]) for d in batch)
        for d in batch:
            d["decoder_input"] = d["decoder_input"] + [word2id["<pad>"]] * (input_maxlen - len(d["decoder_input"]))
            d["decoder_output"] = d["decoder_output"] + [word2id["<pad>"]] * (output_maxlen - len(d["decoder_output"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch], dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch], dtype=torch.long)
        return decoder_inputs, decoder_outputs


def pretrain_gpt(model: GPT, sequences: List[List[int]], device: torch.device,
                 epochs: int = 100, batch_size: int = 128, lr: float = 1e-4,
                 verbose: bool = False) -> GPT:
    """Cross-entropy next-token pretraining of `model` on `sequences`."""
    model = model.to(device)
    dataset = SentenceDataset(sequences)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.padding_batch)

    criterion = nn.CrossEntropyLoss(ignore_index=word2id["<pad>"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for dec_inputs, dec_outputs in loader:
            dec_inputs, dec_outputs = dec_inputs.to(device), dec_outputs.to(device)
            optimizer.zero_grad()
            outputs, _ = model(dec_inputs)
            loss = criterion(outputs, dec_outputs.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            epoch_loss += loss.item()

        if verbose:
            print(f"[EqGPT pretrain] epoch {epoch + 1}/{epochs} loss={epoch_loss / len(loader):.4f}")

    model.eval()
    return model
