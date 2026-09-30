"""Integral-loss symbolic search and a fixed finite-kernel diagnostic."""

import numpy as np

from .kd_dso import KD_DSO


class GreenDSO(KD_DSO):
    def _build_config(self, dataset_arg):
        config = super()._build_config(dataset_arg)
        config["task"]["task_type"] = "green"
        return config

    def fit(self, training_view):
        # Core.setup clears Program.cache and rebinds task/library. Capture the
        # task on the instance: later ordinary DSO fits must not change predictions.
        self._run(training_view)
        if self.best_program_ is None:
            raise ValueError("DSO returned no kernel")
        from .dso.program import Program

        self.task_ = Program.task
        self.execute_function_ = Program.execute_function
        self.protected_ = Program.protected
        self.kernel_shape_ = tuple(training_view["kernel_shape"])
        return self

    def kernel(self, inputs):
        # Program.execute is class-global; temporarily restore this fit's task.
        from .dso.program import Program

        previous_task, previous_library = Program.task, Program.library
        previous_execute, previous_protected = Program.execute_function, Program.protected
        try:
            Program.set_task(self.task_)
            Program.execute_function, Program.protected = self.execute_function_, self.protected_
            result = np.asarray(self.best_program_.execute(inputs), float)
            return np.broadcast_to(result, (len(inputs),)).reshape(self.kernel_shape_).copy()
        finally:
            Program.task, Program.library = previous_task, previous_library
            Program.execute_function, Program.protected = previous_execute, previous_protected
