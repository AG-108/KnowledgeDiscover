"""Adapted from archive/kd/model/DeepSymbolicOptimization/dso/dso/train.py
(original: https://github.com/dso-org/deep-symbolic-optimization).

The risk-seeking training loop itself is framework-agnostic (numpy): quantile
selection, the four baseline variants, Batch assembly, and hall-of-fame
tracking are carried over as-is.

Changes vs. the original
------------------------
- Dropped `tf.compat.v1.logging`, the module-level `tf.set_random_seed(0)`, and
  `sess.run(tf.global_variables_initializer())`; seeding is the caller's job
  (see core.DeepSymbolicOptimizer.set_seeds) and PyTorch modules self-initialize.
- `policy_optimizer.train_step(...)` -> `policy.train_step(...)`; PG and PQT are
  fused into the policy in this port (see policy.py), so the PQT branch is
  selected via `policy.pqt` rather than by the optimizer's concrete type.
- Dropped the `gp_controller` (GP-meld) integration, the `logger`/StatsLogger
  file-output system, and `sample_novel`'s extended-batch path -- none of those
  were ported. `hof`/Pareto tracking is kept but held in memory and exposed as
  attributes instead of being written to disk.
- Kept the `multiprocessing.Pool` reward path (framework-independent), default
  off (n_cores_batch=1).
"""

import time
from itertools import compress

import numpy as np

from .program import Program, from_tokens
from .memory import Batch, make_queue
from .utils import weighted_quantile
from .variance import quantile_variance


# Work for multiprocessing pool: compute reward
def work(p):
    """Compute reward and return it with optimized constants."""
    r = p.r
    return p


def get_duration(start_time):
    return "{:.0f}s".format(time.time() - start_time)


class Trainer:
    """
    Executes the main training loop: sample a batch of expressions from the
    policy, score them, keep the top-epsilon quantile, and take a gradient step.
    """

    def __init__(self, policy, pool=None,
                 n_samples=2000000, batch_size=1000, alpha=0.5,
                 epsilon=0.05, verbose=True, baseline="R_e",
                 b_jumpstart=False, early_stopping=True, debug=0,
                 use_memory=False, memory_capacity=1e3, warm_start=None,
                 memory_threshold=None, complexity="token", hof=100,
                 save_pareto_front=True):
        """
        Parameters
        ----------
        policy : dso.policy.RNNPolicy
            Policy used to generate Programs; also owns the optimizer.

        pool : multiprocessing.Pool or None
            Pool used to parallelize reward computation.

        n_samples : int
            Total number of expressions to sample before finishing.

        batch_size : int
            Number of expressions sampled per iteration.

        alpha : float
            Coefficient of the exponentially-weighted moving average baseline.

        epsilon : float or None
            Fraction of top expressions used for training. None (or 1.0) turns
            off risk-seeking.

        baseline : str
            One of "ewma_R", "R_e", "ewma_R_e", "combined".

        b_jumpstart : bool
            Whether to jump-start the EWMA baseline from the first batch mean.

        early_stopping : bool
            Stop as soon as the task reports success.

        use_memory, memory_capacity, warm_start, memory_threshold :
            Experimental memory-buffer quantile estimation.

        complexity : str
            Complexity measure used for the Pareto front only.

        hof : int
            Size of the hall of fame retained in memory.

        save_pareto_front : bool
            Whether to compute a Pareto front over (complexity, reward).
        """
        self.policy = policy
        self.pool = pool
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.verbose = verbose
        self.baseline = baseline
        self.b_jumpstart = b_jumpstart
        self.early_stopping = early_stopping
        self.debug = debug
        self.use_memory = use_memory
        self.memory_threshold = memory_threshold
        self.complexity = complexity
        self.hof = hof
        self.save_pareto_front = save_pareto_front

        # Priority queue, only when PQT is enabled. Port note: upstream keyed
        # this off `hasattr(policy_optimizer, 'pqt_k')` plus an isinstance check
        # against PQTPolicyOptimizer; here PQT lives on the policy itself.
        if getattr(self.policy, "pqt", False):
            k = self.policy.pqt_k
            self.priority_queue = make_queue(priority=True, capacity=k) \
                if (k is not None and k > 0) else None
        else:
            self.priority_queue = None

        # Memory queue
        if self.use_memory:
            assert self.epsilon is not None and self.epsilon < 1.0, \
                "Memory queue is only used with risk-seeking."
            self.memory_queue = make_queue(policy=self.policy, priority=False,
                                           capacity=int(memory_capacity))
            warm_start = warm_start if warm_start is not None else self.batch_size
            actions, obs, priors = policy.sample(warm_start)
            programs = [from_tokens(a) for a in actions]
            r = np.array([p.r for p in programs])
            l = np.array([len(p.traversal) for p in programs])
            on_policy = np.array([p.originally_on_policy for p in programs])
            sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                                  lengths=l, rewards=r, on_policy=on_policy)
            self.memory_queue.push_batch(sampled_batch, programs)
        else:
            self.memory_queue = None

        self.nevals = 0        # Total number of sampled expressions
        self.iteration = 0
        self.r_best = -np.inf
        self.p_r_best = None
        self.done = False
        self.ewma = None if self.b_jumpstart else 0.0

        # In-memory replacements for the dropped file logger
        self.hall_of_fame = []
        self.history = []

    def _update_hall_of_fame(self, programs):
        """Keep the top-`hof` unique programs by reward, in memory."""
        if not self.hof:
            return
        seen = {p.str: p for p in self.hall_of_fame}
        for p in programs:
            if p.str not in seen and np.isfinite(p.r):
                seen[p.str] = p
        ranked = sorted(seen.values(), key=lambda p: p.r, reverse=True)
        self.hall_of_fame = ranked[:self.hof]

    def pareto_front(self):
        """Pareto-optimal programs over (complexity, reward), lower complexity
        and higher reward being better."""
        if not self.hall_of_fame:
            return []
        pool = sorted(self.hall_of_fame, key=lambda p: (p.complexity, -p.r))
        front, best_r = [], -np.inf
        for p in pool:
            if p.r > best_r:
                front.append(p)
                best_r = p.r
        return front

    def run_one_step(self, override=None):
        """
        Execute one iteration of the training loop. If `override` is given,
        train on that (actions, obs, priors, programs) tuple instead of
        sampling.
        """
        start_time = time.time()

        if override is None:
            actions, obs, priors = self.policy.sample(self.batch_size)
            programs = [from_tokens(a) for a in actions]
        else:
            actions, obs, priors, programs = override
            for p in programs:
                Program.cache[p.str] = p

        self.nevals += self.batch_size

        # Compute rewards in parallel
        if self.pool is not None:
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
            pool_p_dict = {p.str: p for p in self.pool.map(work, programs_to_optimize)}
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__ else p for p in programs]
            Program.cache.update(pool_p_dict)

        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r for p in programs])
        r_full = r.copy()

        l = np.array([len(p.traversal) for p in programs])
        on_policy = np.array([p.originally_on_policy for p in programs])
        invalid = np.array([p.invalid for p in programs], dtype=bool)

        r_max = np.nanmax(r) if np.isfinite(r).any() else -np.inf

        # ---- Risk-seeking: keep the top-epsilon quantile ----
        if self.epsilon is not None and self.epsilon < 1.0:
            if self.memory_queue is not None:
                memory_r = self.memory_queue.get_rewards()
                memory_w = self.memory_queue.compute_probs()
                if len(memory_r) < self.memory_queue.capacity:
                    combined_r = r
                    combined_w = np.repeat(1.0 / len(r), len(r))
                else:
                    unique_programs = [p for p in programs
                                       if p.str not in self.memory_queue.unique_items]
                    N = len(unique_programs)
                    combined_r = np.concatenate([memory_r, r])
                    if N == 0:
                        combined_w = memory_w / memory_w.sum()
                    else:
                        sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                        combined_w = np.concatenate([memory_w, sample_w])
                    if self.memory_threshold is not None and memory_w.sum() > self.memory_threshold:
                        quantile_variance(self.memory_queue, self.policy,
                                          self.batch_size, self.epsilon, self.iteration)
                quantile = weighted_quantile(values=combined_r, weights=combined_w,
                                             q=1 - self.epsilon)
            else:
                # Port note: numpy>=1.22 renamed the `interpolation` kwarg to
                # `method`; upstream passed interpolation="higher".
                quantile = np.quantile(r, 1 - self.epsilon, method="higher")

            keep = r >= quantile
            l = l[keep]
            invalid = invalid[keep]
            r = r[keep]
            programs = list(compress(programs, keep))
            actions = actions[keep, :]
            obs = obs[keep, :, :]
            priors = priors[keep, :, :]
            on_policy = on_policy[keep]
        else:
            quantile = np.nan

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)

        # ---- Baseline ----
        if self.baseline == "ewma_R":
            self.ewma = np.mean(r) if self.ewma is None \
                else self.alpha * np.mean(r) + (1 - self.alpha) * self.ewma
            b = self.ewma
        elif self.baseline == "R_e":  # Default
            self.ewma = -1
            b = quantile
        elif self.baseline == "ewma_R_e":
            self.ewma = np.min(r) if self.ewma is None \
                else self.alpha * quantile + (1 - self.alpha) * self.ewma
            b = self.ewma
        elif self.baseline == "combined":
            val = np.mean(r) - quantile
            self.ewma = val if self.ewma is None \
                else self.alpha * val + (1 - self.alpha) * self.ewma
            b = quantile + self.ewma
        else:
            raise ValueError(f"Unknown baseline: {self.baseline}")

        lengths = np.array([min(len(p.traversal), self.policy.max_length)
                            for p in programs], dtype=np.int32)

        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=lengths, rewards=r, on_policy=on_policy)

        # ---- Gradient step ----
        if self.priority_queue is not None:
            self.priority_queue.push_best(sampled_batch, programs)
            pqt_batch = self.priority_queue.sample_batch(self.policy.pqt_batch_size)
            loss = self.policy.train_step(b, sampled_batch, pqt_batch)
        else:
            loss = self.policy.train_step(b, sampled_batch)

        if self.memory_queue is not None:
            self.memory_queue.push_batch(sampled_batch, programs)

        # ---- Track best ----
        if r_max > self.r_best:
            self.r_best = r_max
            self.p_r_best = programs[np.argmax(r)] if len(programs) else self.p_r_best
            if self.verbose or self.debug:
                print("[{}] Training iteration {}, current best R: {:.4f}".format(
                    get_duration(start_time), self.iteration + 1, self.r_best))

        self._update_hall_of_fame(programs)
        self.history.append({
            "iteration": self.iteration + 1,
            "r_best": float(self.r_best),
            "r_max": float(r_max) if np.isfinite(r_max) else None,
            "r_avg_full": float(np.nanmean(r_full)) if np.isfinite(r_full).any() else None,
            "baseline": float(b) if np.isfinite(b) else None,
            "loss": float(loss),
            "nevals": self.nevals,
            "walltime": time.time() - start_time,
        })

        # ---- Stopping ----
        if self.early_stopping and self.p_r_best is not None \
                and self.p_r_best.evaluate.get("success"):
            if self.verbose:
                print("[{}] Early stopping criteria met; breaking early.".format(
                    get_duration(start_time)))
            self.done = True

        if self.verbose and (self.iteration + 1) % 10 == 0:
            print("[{}] Training iteration {}, current best R: {:.4f}".format(
                get_duration(start_time), self.iteration + 1, self.r_best))

        if self.nevals >= self.n_samples:
            self.done = True

        self.iteration += 1
