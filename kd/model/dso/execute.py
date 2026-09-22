"""Adapted from archive/kd/model/DeepSymbolicOptimization/dso/dso/execute.py
(original: https://github.com/dso-org/deep-symbolic-optimization).

Changes vs. the original: `from dso import cyfunc` -> `from . import cyfunc`.
The compiled Cython extension is not built in this port, so cyfunc stays None
and Program.set_execute falls back to python_execute.
"""
try:
    # NOTE (port): was `from dso import cyfunc` upstream. `cyfunc` is an
    # optional compiled Cython extension; it is not built in this port, so this
    # import fails and `cyfunc` stays None -- see the guard in
    # Program.set_execute, which selects python_execute in that case.
    from . import cyfunc
except ImportError:
    cyfunc = None
import array

from .library import StateChecker, Polynomial


def python_execute(traversal, X):
    """
    Executes the program according to X using Python.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    Returns
    -------
    y_hats : array-like, shape = [n_samples]
        The result of executing the program on X.
    """

    apply_stack = []

    for node in traversal:
        apply_stack.append([node])

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]

            if token.input_var is not None:
                intermediate_result = X[:, token.input_var]
            else:
                if isinstance(token, StateChecker):
                    token.set_state_value(X[:, token.state_index])
                if isinstance(token, Polynomial):
                    intermediate_result = token(X)
                else:
                    intermediate_result = token(*terminals)
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result

    assert False, "Function should never get here!"
    return None

def cython_execute(traversal, X):
    """
    Execute cython function using given traversal over input X.

    Parameters
    ----------

    traversal : list
        A list of nodes representing the traversal over a Program.
    X : np.array
        The input values to execute the traversal over.

    Returns
    -------

    result : float
        The result of executing the traversal.
    """
    if len(traversal) > 1:
        is_input_var = array.array('i', [t.input_var is not None for t in traversal])
        return cyfunc.execute(X, len(traversal), traversal, is_input_var)
    else:
        return python_execute(traversal, X)


