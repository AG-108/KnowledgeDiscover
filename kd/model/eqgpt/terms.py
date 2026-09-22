"""
Symbolic-term evaluation via autograd on the trained surrogate network.

Adapted from `kd/dataset/EqGPT/code/calculate_terms.py` (original EqGPT
repo). Only entries that actually occur in this package's vendored
vocabulary (`kd/model/eqgpt/data/dict_datas_0725.json`, 57 tokens) are
ported. One correctness fix vs. the original: the term "sint" (meant to be
sin(t)) mistakenly evaluated `sin(x)` in the source (using `x_index`
instead of `t_index`) -- fixed here.
"""

from typing import List

import numpy as np
import torch


def calculate_terms(word: str, net: torch.nn.Module, database: torch.Tensor,
                    variables: List[str]) -> np.ndarray:
    """
    Evaluate a single vocabulary term (e.g. "uxxx", "sin(u)", "x") on
    `database` (an (N, len(variables)) tensor with requires_grad=True),
    using `net` as the surrogate u(variables...) and `torch.autograd.grad`
    for spatial/temporal derivatives. Returns a flat numpy array.
    """
    x_index = variables.index("x") if "x" in variables else None
    t_index = variables.index("t") if "t" in variables else None
    y_index = variables.index("y") if "y" in variables else None
    z_index = variables.index("z") if "z" in variables else None

    def grad(output, idx):
        return torch.autograd.grad(outputs=output.sum(), inputs=database, create_graph=True)[0][:, idx]

    u = net(database)
    h_grad = torch.autograd.grad(outputs=u.sum(), inputs=database, create_graph=True)[0]
    hx = h_grad[:, x_index].reshape(-1, 1) if x_index is not None else None
    ht = h_grad[:, t_index].reshape(-1, 1) if t_index is not None else None
    hy = h_grad[:, y_index].reshape(-1, 1) if y_index is not None else None
    hz = h_grad[:, z_index].reshape(-1, 1) if z_index is not None else None

    def np_(t):
        return t.cpu().data.numpy()

    if word == "u":
        return np_(u)
    if word == "ut":
        return np_(ht)
    if word == "ux":
        return np_(hx)
    if word == "uxx":
        return np_(grad(hx, x_index))
    if word == "uxxt":
        hxx = grad(hx, x_index).reshape(-1, 1)
        return np_(grad(hxx, t_index))
    if word == "ut^2":
        return np_(ht ** 2)
    if word == "uxt":
        return np_(grad(hx, t_index))
    if word == "ux^2":
        return np_(hx ** 2)
    if word == "utt":
        return np_(grad(ht, t_index))
    if word == "uxxxx":
        hxx = grad(hx, x_index).reshape(-1, 1)
        hxxx = grad(hxx, x_index).reshape(-1, 1)
        return np_(grad(hxxx, x_index))
    if word == "uxxxxx":
        hxx = grad(hx, x_index).reshape(-1, 1)
        hxxx = grad(hxx, x_index).reshape(-1, 1)
        hxxxx = grad(hxxx, x_index).reshape(-1, 1)
        return np_(grad(hxxxx, x_index))
    if word == "uxxx":
        hxx = grad(hx, x_index).reshape(-1, 1)
        return np_(grad(hxx, x_index))
    if word == "(u^2)xx":
        h2x = grad(hx ** 2, x_index).reshape(-1, 1)
        return np_(grad(h2x, x_index))
    if word == "(uux)x":
        return np_(grad(u * hx, x_index))
    if word == "(uux)t":
        return np_(grad(u * hx, t_index))
    if word == "(uux)xx":
        temp = grad(u * hx, x_index).reshape(-1, 1)
        return np_(grad(temp, x_index))
    if word == "uxxtt":
        hxx = grad(hx, x_index).reshape(-1, 1)
        hxxt = grad(hxx, t_index).reshape(-1, 1)
        return np_(grad(hxxt, t_index))
    if word == "(u^4)xx":
        h4x = grad(u ** 4, x_index).reshape(-1, 1)
        return np_(grad(h4x, x_index))
    if word == "(u^3)xx":
        h3x = grad(u ** 3, x_index).reshape(-1, 1)
        return np_(grad(h3x, x_index))
    if word == "u^2":
        return np_(u ** 2)
    if word == "u^3":
        return np_(u ** 3)
    if word == "x":
        return np_(database[:, x_index])
    if word == "t":
        return np_(database[:, t_index])
    if word == "(1/u)xx":
        temp = grad(u ** (-1), x_index).reshape(-1, 1)
        return np_(grad(temp, x_index))
    if word == "(u^-2*ux)x":
        return np_(grad(u ** (-2) * hx, x_index))
    if word == "ut^3":
        return np_(ht ** 3)
    if word == "sqrt(u)":
        return np_(torch.sqrt(u))
    if word == "sin(u)":
        return np_(torch.sin(u))
    if word == "sinh(u)":
        return np_(torch.sinh(u))
    if word == "x^2":
        return np_(database[:, x_index] ** 2)
    if word == "x^4":
        return np_(database[:, x_index] ** 4)
    if word == "sqrt(x)":
        return np_(torch.sqrt(database[:, x_index]))
    if word == "exp(x)":
        return np_(torch.exp(database[:, x_index]))
    if word == "sint":
        return np_(torch.sin(database[:, t_index]))  # fixed: original used x_index
    if word == "sinx":
        return np_(torch.sin(database[:, x_index]))
    if word == "(uxx+ux/x)^2":
        hxx = grad(hx, x_index).reshape(-1, 1)
        return np_((hxx + (1.0 / database[:, x_index].reshape(-1, 1)) * hx) ** 2)
    if word == "(u(u^2)xx)xx":
        temp = grad(u * hx, x_index).reshape(-1, 1)
        temp1 = grad(temp, x_index).reshape(-1, 1)
        temp2 = grad(u * temp1, x_index).reshape(-1, 1)
        return np_(grad(temp2, x_index))

    # -- multi-dimensional terms (y/z), only reachable if `variables`
    #    includes y/z; masked out for the 1D (x, t) KdV example --
    if word == "y":
        return np_(database[:, y_index])
    if word == "y^2":
        return np_(database[:, y_index] ** 2)
    if word == "uy":
        return np_(hy)
    if word == "uy^2":
        return np_(hy ** 2)
    if word == "uyy":
        return np_(grad(hy, y_index))
    if word == "uyyy":
        hyy = grad(hy, y_index).reshape(-1, 1)
        return np_(grad(hyy, y_index))
    if word == "uyyt":
        hyy = grad(hy, y_index).reshape(-1, 1)
        return np_(grad(hyy, t_index))
    if word == "uxy":
        return np_(grad(hx, y_index))
    if word == "uz":
        return np_(hz)
    if word == "uzz":
        return np_(grad(hz, z_index))
    if word == "(x+y)":
        return np_(database[:, x_index] + database[:, y_index])
    if word == "exp(-y)":
        return np_(torch.exp(-database[:, y_index]))
    if word == "Laplace(u)":
        hxx = grad(hx, x_index).reshape(-1, 1)
        hyy = grad(hy, y_index).reshape(-1, 1)
        if z_index is not None:
            hzz = grad(hz, z_index).reshape(-1, 1)
            return np_(hxx + hyy + hzz)
        return np_(hxx + hyy)
    if word == "BiLaplace(u)":
        hxx = grad(hx, x_index).reshape(-1, 1)
        hxxx = grad(hxx, x_index).reshape(-1, 1)
        hxxxx = grad(hxxx, x_index).reshape(-1, 1)
        hyy = grad(hy, y_index).reshape(-1, 1)
        hyyy = grad(hyy, y_index).reshape(-1, 1)
        hyyyy = grad(hyyy, y_index).reshape(-1, 1)
        hxxy = grad(hxx, y_index).reshape(-1, 1)
        hxxyy = grad(hxxy, y_index).reshape(-1, 1)
        return np_(hxxxx + hyyyy + 2 * hxxyy)
    if word == "Laplace(utt)":
        utt = grad(ht, t_index)
        uttx = grad(utt, x_index)
        uttxx = grad(uttx, x_index)
        utty = grad(utt, y_index)
        uttyy = grad(utty, y_index)
        if z_index is not None:
            uttz = grad(utt, z_index)
            uttzz = grad(uttz, z_index)
            return np_(uttxx + uttyy + uttzz)
        return np_(uttxx + uttyy)

    raise KeyError(f"Unsupported EqGPT vocabulary term: {word!r}")
