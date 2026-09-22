"""Adapted from archive/kd/model/DeepSymbolicOptimization/dso/dso/tf_state_manager.py
(original: https://github.com/dso-org/deep-symbolic-optimization), ported from
TensorFlow 1.x to PyTorch.

Changes vs. the original:
- tf.one_hot -> F.one_hot; tf.nn.embedding_lookup on a tf.get_variable ->
  indexing an nn.Parameter; tf.unstack/tf.concat -> torch splits/torch.cat.
- Embeddings are returned via `parameters()` so the owning policy can register
  them with its optimizer (TF tracked them implicitly through the graph).
- `get_tensor_input` accepts either a numpy array or a torch tensor and always
  returns a float32 tensor on the policy's device.

The PyTorch idioms here follow kd/model/discover/state_manager.py, but the
parameter semantics are the DSO originals (observe_action/parent/sibling/
dangling + embedding/embedding_size); DISCOVER's extra GrammarStateManager is
not carried over.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .program import Program


class StateManager(ABC):
    """
    An interface for handling the torch.Tensor inputs to the Policy.
    """

    def setup_manager(self, policy):
        """
        Function called inside the policy to perform needed initializations.

        Parameters
        ----------
        policy : the policy object that owns this state manager.
        """
        self.policy = policy
        self.max_length = policy.max_length

    @abstractmethod
    def get_tensor_input(self, obs):
        """
        Convert an observation from a Task into a Tensor input for the Policy,
        e.g. by performing one-hot encoding or embedding lookup.

        Parameters
        ----------
        obs : np.ndarray (dtype=np.float32) or torch.Tensor
            Observation coming from the Task, shape (batch, OBS_DIM).

        Returns
        -------
        input_ : torch.Tensor (dtype=torch.float32)
            Tensor to be used as input to the Policy.
        """
        return

    def process_state(self, obs):
        """
        Entry point for adding information to the state tuple.
        If not overwritten, this function does nothing.
        """
        return obs

    def parameters(self):
        """Trainable parameters owned by this state manager (embeddings, if
        enabled). Returns an empty list when there are none.

        Port note: under TF these were graph variables picked up automatically
        by the optimizer; in PyTorch the owning policy must be handed them
        explicitly.
        """
        return []


def make_state_manager(config):
    """
    Parameters
    ----------
    config : dict
        Parameters for this StateManager.

    Returns
    -------
    state_manager : StateManager
        The StateManager to be used by the policy.
    """
    manager_dict = {
        "hierarchical": HierarchicalStateManager
    }

    if config is None:
        config = {}
    else:
        config = dict(config)  # don't mutate the caller's config

    # Use HierarchicalStateManager by default
    manager_type = config.pop("type", "hierarchical")

    manager_class = manager_dict[manager_type]
    state_manager = manager_class(**config)

    return state_manager


class HierarchicalStateManager(StateManager):
    """
    Class that uses the previous action, parent, sibling, and/or dangling as
    observations.
    """

    def __init__(self, observe_parent=True, observe_sibling=True,
                 observe_action=False, observe_dangling=False, embedding=False,
                 embedding_size=8):
        """
        Parameters
        ----------
        observe_parent : bool
            Observe the parent of the Token being selected?

        observe_sibling : bool
            Observe the sibling of the Token being selected?

        observe_action : bool
            Observe the previously selected Token?

        observe_dangling : bool
            Observe the number of dangling nodes?

        embedding : bool
            Use embeddings for categorical inputs?

        embedding_size : int
            Size of embeddings for each categorical input if embedding=True.
        """
        self.observe_parent = observe_parent
        self.observe_sibling = observe_sibling
        self.observe_action = observe_action
        self.observe_dangling = observe_dangling
        self.library = Program.library

        # Parameter assertions/warnings
        assert self.observe_action + self.observe_parent + self.observe_sibling + self.observe_dangling > 0, \
            "Must include at least one observation."

        self.embedding = embedding
        self.embedding_size = embedding_size

        self._embeddings = []

        # Input width the policy's RNN should expect. Computed here so the
        # policy does not have to re-derive it (TF discovered this by tracing a
        # dummy tensor through the graph).
        width = 0
        if self.observe_action:
            width += embedding_size if embedding else self.library.n_action_inputs
        if self.observe_parent:
            width += embedding_size if embedding else self.library.n_parent_inputs
        if self.observe_sibling:
            width += embedding_size if embedding else self.library.n_sibling_inputs
        if self.observe_dangling:
            width += 1
        self.input_dim = width

    def setup_manager(self, policy):
        super().setup_manager(policy)
        device = getattr(policy, "device", torch.device("cpu"))

        # Create embeddings if needed
        if self.embedding:
            def _make(n):
                p = nn.Parameter(torch.empty(n, self.embedding_size, device=device))
                nn.init.uniform_(p, a=-1.0, b=1.0)
                self._embeddings.append(p)
                return p

            if self.observe_action:
                self.action_embeddings = _make(self.library.n_action_inputs)
            if self.observe_parent:
                self.parent_embeddings = _make(self.library.n_parent_inputs)
            if self.observe_sibling:
                self.sibling_embeddings = _make(self.library.n_sibling_inputs)

    def parameters(self):
        return self._embeddings

    def get_tensor_input(self, obs):
        device = getattr(self.policy, "device", torch.device("cpu")) \
            if hasattr(self, "policy") else torch.device("cpu")

        if not torch.is_tensor(obs):
            obs = torch.as_tensor(np.asarray(obs, dtype=np.float32))
        obs = obs.to(device=device, dtype=torch.float32)

        observations = []
        # obs is (batch, OBS_DIM); first four columns are the hierarchical state
        action = obs[:, 0].long()
        parent = obs[:, 1].long()
        sibling = obs[:, 2].long()
        dangling = obs[:, 3]

        # Action, parent, and sibling inputs are either one-hot or embeddings
        if self.observe_action:
            if self.embedding:
                x = self.action_embeddings[action]
            else:
                x = F.one_hot(action, num_classes=self.library.n_action_inputs).float()
            observations.append(x)
        if self.observe_parent:
            if self.embedding:
                x = self.parent_embeddings[parent]
            else:
                x = F.one_hot(parent, num_classes=self.library.n_parent_inputs).float()
            observations.append(x)
        if self.observe_sibling:
            if self.embedding:
                x = self.sibling_embeddings[sibling]
            else:
                x = F.one_hot(sibling, num_classes=self.library.n_sibling_inputs).float()
            observations.append(x)

        # Dangling input is just the value of dangling
        if self.observe_dangling:
            observations.append(dangling.unsqueeze(-1))

        input_ = torch.cat(observations, dim=-1)

        # Possibly concatenate additional observations beyond the first four
        # (the original supported task-supplied extras here).
        if obs.shape[1] > 4:
            input_ = torch.cat([input_, obs[:, 4:]], dim=-1)

        return input_
