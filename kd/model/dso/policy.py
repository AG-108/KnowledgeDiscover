"""Adapted from archive/kd/model/DeepSymbolicOptimization/dso/dso/policy/{policy,rnn_policy}.py
plus policy_optimizer/{policy_optimizer,pg_policy_optimizer,pqt_policy_optimizer}.py
(original: https://github.com/dso-org/deep-symbolic-optimization), ported from
TensorFlow 1.x to PyTorch.

Changes vs. the original
------------------------
- `tf.nn.raw_rnn` + `loop_fn` autoregressive sampling -> an explicit Python
  timestep loop over `nn.LSTM`/`nn.GRU` (one step at a time, carrying hidden
  state). The `tf.py_func` call into `task.get_next_obs` becomes a direct call.
- `tf.nn.dynamic_rnn` (used to re-score a stored batch in
  `make_neglogp_and_entropy`) -> a single batched forward pass over the whole
  padded sequence, with an explicit length mask.
- `tf.placeholder`/`feed_dict`/`sess.run` -> ordinary tensor arguments.
- The Policy/PolicyOptimizer split (three optimizer subclasses over a shared
  ABC) existed to accommodate TF's graph-construction flow. Following
  kd/model/discover/controller.py, PG and PQT are collapsed into this one
  class: both losses are assembled in `train_step`. PPO was not ported (marked
  EXPERIMENTAL and commented out in the original's own configs).
- `tf.summary` logging dropped.

Semantics preserved from the original: risk-seeking policy gradient
(`(r - baseline) * neglogp`), the entropy bonus with `entropy_gamma**t` decay,
`safe_cross_entropy`'s guard against `0 * -inf`, additive prior logits, and the
optional `action_prob_lowerbound` mixing.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .program import Program
from .memory import Batch


def safe_cross_entropy(p, logq, dim=-1):
    """Compute p * logq safely, by susbtituting logq -> 0 wherever p == 0.

    Port note: identical in intent to the original's tf version -- prevents
    0 * -inf = nan when the prior has hard-masked an action.
    """
    safe_logq = torch.where(p == 0, torch.zeros_like(logq), logq)
    return -torch.sum(p * safe_logq, dim=dim)


class RNNPolicy(nn.Module):
    """
    Recurrent neural network (RNN) policy used to generate expressions.

    The RNN outputs a distribution over pre-order traversals of symbolic
    expression trees. It is trained with REINFORCE plus a baseline
    (risk-seeking by default), optionally combined with priority queue
    training (PQT).

    Parameters
    ----------
    prior : dso.prior.JointPrior
        JointPrior used to constrain/adjust probabilities during sampling.

    state_manager : dso.state_manager.StateManager
        Handles the conversion of observations into RNN inputs.

    max_length : int
        Maximum sequence (traversal) length.

    cell : str
        Recurrent cell to use; 'lstm' or 'gru'.

    num_layers : int
        Number of RNN layers.

    num_units : int
        Number of hidden units per RNN layer.

    initializer : str
        Weight initializer for the recurrent cell; 'zeros' or 'var_scale'.

    optimizer : str
        Optimizer name; 'adam', 'rmsprop', or 'sgd'.

    learning_rate : float
        Optimizer learning rate.

    entropy_weight : float
        Coefficient of the entropy bonus.

    entropy_gamma : float
        Per-timestep decay applied to the entropy bonus (`entropy_gamma**t`).

    policy_optimizer_type : str
        'pg' (policy gradient) or 'pqt' (priority queue training).

    pqt_k, pqt_batch_size, pqt_weight, pqt_use_pg :
        PQT hyperparameters; only read when policy_optimizer_type == 'pqt'.

    action_prob_lowerbound : float
        If nonzero, mix each action distribution with a uniform distribution by
        this weight before applying the prior.

    device : torch.device
        Device to place the policy on.
    """

    def __init__(self, prior, state_manager,
                 max_length=64,
                 cell='lstm',
                 num_layers=1,
                 num_units=32,
                 initializer='zeros',
                 optimizer='adam',
                 learning_rate=0.0005,
                 entropy_weight=0.03,
                 entropy_gamma=0.7,
                 policy_optimizer_type='pg',
                 pqt_k=10,
                 pqt_batch_size=1,
                 pqt_weight=200.0,
                 pqt_use_pg=False,
                 action_prob_lowerbound=0.0,
                 debug=0,
                 device=None):
        super().__init__()

        assert 0 <= action_prob_lowerbound <= 1
        self.action_prob_lowerbound = action_prob_lowerbound

        self.prior = prior
        self.state_manager = state_manager
        self.max_length = max_length
        self.n_choices = Program.library.L
        self.debug = debug
        self.device = device if device is not None else torch.device("cpu")

        self.entropy_weight = entropy_weight
        self.entropy_gamma = entropy_gamma

        self.policy_optimizer_type = policy_optimizer_type
        self.pqt = (policy_optimizer_type == 'pqt')
        self.pqt_k = pqt_k
        self.pqt_batch_size = pqt_batch_size
        self.pqt_weight = pqt_weight
        self.pqt_use_pg = pqt_use_pg

        # Let the state manager build any parameters it owns (embeddings) and
        # learn our max_length / device.
        self.state_manager.setup_manager(self)

        input_dim = getattr(self.state_manager, "input_dim", None)
        assert input_dim is not None, \
            "state_manager must expose input_dim (width of get_tensor_input output)"
        self.input_dim = input_dim

        # Recurrent cell
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.cell_type = cell
        self.num_layers = num_layers
        self.num_units = num_units
        self.rnn = rnn_cls(input_size=input_dim, hidden_size=num_units,
                           num_layers=num_layers, batch_first=True)
        # LinearWrapper in the original: projects cell output -> n_choices
        self.head = nn.Linear(num_units, self.n_choices)

        self._apply_initializer(initializer)
        self.to(self.device)

        # Entropy decay vector, precomputed as in the original
        gamma = 1.0 if entropy_gamma is None else entropy_gamma
        self.register_buffer(
            "entropy_gamma_decay",
            torch.tensor(
                [gamma ** t for t in range(max_length)],
                dtype=torch.float32,
                device=self.device,
            ),
        )

        # Optimizer over RNN + head + any state-manager embeddings
        params = list(self.parameters()) + list(self.state_manager.parameters())
        opt_cls = {"adam": torch.optim.Adam,
                   "rmsprop": torch.optim.RMSprop,
                   "sgd": torch.optim.SGD}[optimizer]
        self.optimizer = opt_cls(params, lr=learning_rate)

    def _apply_initializer(self, initializer):
        """Port note: the original only implemented 'zeros' in make_initializer
        (var_scale was named in the docstring but unreachable); both are
        supported here, zeros remaining the default."""
        if initializer == "zeros":
            init_fn = nn.init.zeros_
        elif initializer == "var_scale":
            init_fn = lambda w: nn.init.kaiming_uniform_(w, a=np.sqrt(5), mode="fan_avg") \
                if w.dim() > 1 else nn.init.zeros_(w)
        else:
            raise ValueError(f"Unknown initializer: {initializer}")

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                init_fn(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def _zero_state(self, batch_size):
        h = torch.zeros(self.num_layers, batch_size, self.num_units, device=self.device)
        if self.cell_type == "lstm":
            c = torch.zeros(self.num_layers, batch_size, self.num_units, device=self.device)
            return (h, c)
        return h

    def apply_action_prob_lowerbound(self, logits):
        """Applies a lower bound to the probability of each action."""
        probs = F.softmax(logits, dim=-1)
        probs_bounded = ((1 - self.action_prob_lowerbound) * probs
                         + self.action_prob_lowerbound / float(self.n_choices))
        return torch.log(probs_bounded)

    @torch.no_grad()
    def sample(self, n):
        """
        Sample n expressions autoregressively.

        Returns
        -------
        actions : np.ndarray (int32), shape (n, max_length)
        obs : np.ndarray (float32), shape (n, OBS_DIM, max_length)
        priors : np.ndarray (float32), shape (n, max_length, n_choices)

        Port note: replaces tf.nn.raw_rnn's loop_fn. The original emitted
        TensorArrays inside the graph; here each timestep appends to a Python
        list and we stack at the end.
        """
        self.eval()
        task = Program.task

        # Initial observation and prior, per the original's time == 0 branch
        initial_obs = task.reset_task(self.prior)          # (OBS_DIM,)
        initial_obs = np.broadcast_to(initial_obs, (n, task.OBS_DIM)).astype(np.float32).copy()
        obs = self.state_manager.process_state(initial_obs)

        initial_prior = self.prior.initial_prior()          # (n_choices,)
        prior = np.broadcast_to(initial_prior, (n, self.n_choices)).astype(np.float32).copy()

        finished = np.zeros(n, dtype=bool)
        cell_state = self._zero_state(n)

        actions_l, obs_l, priors_l = [], [], []

        for t in range(self.max_length):
            # Record the observation/prior the action is about to be drawn
            # under (the original wrote OLD obs/prior at each step).
            obs_l.append(obs.copy())
            priors_l.append(prior.copy())

            inp = self.state_manager.get_tensor_input(obs).unsqueeze(1)  # (n, 1, input_dim)
            cell_out, cell_state = self.rnn(inp, cell_state)
            logits = self.head(cell_out[:, 0, :])                        # (n, n_choices)

            if self.action_prob_lowerbound != 0.0:
                logits = self.apply_action_prob_lowerbound(logits)

            logits = logits + torch.as_tensor(prior, device=self.device)

            action = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)[:, 0]
            action_np = action.detach().cpu().numpy().astype(np.int32)
            actions_l.append(action_np)

            actions_so_far = np.stack(actions_l, axis=1)                 # (n, t+1)
            next_obs, next_prior, next_finished = task.get_next_obs(
                actions_so_far, obs, finished)

            obs = self.state_manager.process_state(next_obs.astype(np.float32))
            prior = next_prior.astype(np.float32)
            finished = np.logical_or(next_finished, t + 1 >= self.max_length)

            if finished.all():
                break

        actions = np.stack(actions_l, axis=1).astype(np.int32)           # (n, T)
        obs_arr = np.stack(obs_l, axis=2).astype(np.float32)             # (n, OBS_DIM, T)
        priors_arr = np.stack(priors_l, axis=1).astype(np.float32)       # (n, T, n_choices)

        return actions, obs_arr, priors_arr

    def make_neglogp_and_entropy(self, B):
        """
        Compute negative log-probabilities and entropy for a stored Batch under
        the current policy.

        Parameters
        ----------
        B : Batch
            Batch with fields actions, obs, priors, lengths, rewards, on_policy.

        Returns
        -------
        neglogp : torch.Tensor, shape (batch,)
        entropy : torch.Tensor, shape (batch,)
        """
        actions = torch.as_tensor(np.asarray(B.actions), dtype=torch.long, device=self.device)
        priors = torch.as_tensor(np.asarray(B.priors), dtype=torch.float32, device=self.device)
        lengths = torch.as_tensor(np.asarray(B.lengths), dtype=torch.long, device=self.device)
        obs = np.asarray(B.obs)

        batch, B_max_length = actions.shape

        # obs arrives as (batch, OBS_DIM, time); the state manager consumes
        # (rows, OBS_DIM), so flatten time into the row axis and restore after.
        obs_t = np.transpose(obs, (0, 2, 1)).reshape(-1, obs.shape[1])
        inputs = self.state_manager.get_tensor_input(obs_t)
        inputs = inputs.view(batch, B_max_length, self.input_dim)

        # Port note: the original used tf.nn.dynamic_rnn with sequence_length so
        # gradients stopped past each sequence's end. Here we run the full
        # padded sequence and rely on the length mask below; positions beyond
        # the length contribute 0 to both sums, so gradients match.
        cell_out, _ = self.rnn(inputs, self._zero_state(batch))
        logits = self.head(cell_out)

        if self.action_prob_lowerbound != 0.0:
            logits = self.apply_action_prob_lowerbound(logits)

        logits = logits + priors
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)

        # Mask from sequence lengths
        ar = torch.arange(B_max_length, device=self.device).unsqueeze(0)
        mask = (ar < lengths.unsqueeze(1)).float()

        actions_one_hot = F.one_hot(actions, num_classes=self.n_choices).float()
        neglogp_per_step = safe_cross_entropy(actions_one_hot, logprobs, dim=2)
        neglogp = torch.sum(neglogp_per_step * mask, dim=1)

        # If entropy_gamma == 1, entropy_gamma_decay_mask == mask
        decay = self.entropy_gamma_decay[:B_max_length].unsqueeze(0)
        entropy_gamma_decay_mask = decay * mask
        entropy_per_step = safe_cross_entropy(probs, logprobs, dim=2)
        entropy = torch.sum(entropy_per_step * entropy_gamma_decay_mask, dim=1)

        return neglogp, entropy

    def train_step(self, baseline, sampled_batch, pqt_batch=None):
        """
        One gradient step.

        Port note: this fuses what the original split across
        PolicyOptimizer._init_loss_with_entropy (entropy term),
        PGPolicyOptimizer._set_loss (policy-gradient term) and
        PQTPolicyOptimizer._set_loss (priority-queue term), following
        kd/model/discover/controller.py:train_step.

        Returns
        -------
        loss : float
        """
        self.train()
        self.optimizer.zero_grad()

        neglogp, entropy = self.make_neglogp_and_entropy(sampled_batch)
        r = torch.as_tensor(np.asarray(sampled_batch.rewards),
                            dtype=torch.float32, device=self.device)

        # Entropy bonus (negated: we minimize loss)
        loss = -self.entropy_weight * torch.mean(entropy)

        if (not self.pqt) or (self.pqt and self.pqt_use_pg):
            b = torch.as_tensor(float(baseline), dtype=torch.float32, device=self.device)
            loss = loss + torch.mean((r - b) * neglogp)

        if self.pqt:
            assert pqt_batch is not None, "PQT enabled but no pqt_batch supplied"
            pqt_neglogp, _ = self.make_neglogp_and_entropy(pqt_batch)
            loss = loss + self.pqt_weight * torch.mean(pqt_neglogp)

        loss.backward()
        self.optimizer.step()

        return float(loss.detach().cpu())

    @torch.no_grad()
    def compute_probs(self, memory_batch, log=False):
        """Probability (or log-probability) of each sequence in a Batch.

        Used by the memory-buffer / quantile-variance machinery.
        """
        neglogp, _ = self.make_neglogp_and_entropy(memory_batch)
        logps = -neglogp
        out = logps if log else torch.exp(logps)
        return out.detach().cpu().numpy()
