"""Random equation generator, adapted from
archive/kd/model/SymbolicGPT/generator/treeBased/generateData.py (flattened
out of its `generator/treeBased/` nesting since it's the only generator used).

Changes vs. the original:
- The original module reseeded the *global* `random`/`numpy` RNGs to a
  hardcoded `2021` as a side effect of merely importing it (with a comment
  saying to hand-edit that constant to 2022/2023 for val/test splits) --
  this silently clobbers whatever random state any other code in the same
  process was relying on. Removed: callers are expected to seed
  `random`/`numpy` themselves before calling `generate_equation` (this is
  what `kd_symbolicgpt.py`'s `fit()` already does for every other seeded
  step, matching the convention used by `KD_EqGPT.fit()`).
- Dropped `raw_eqn_to_skeleton_structure`/`eqn_to_str_skeleton_structure`
  (an alternate structural-skeleton builder that `dataGen` never actually
  calls -- it uses the regex-based `eqn_to_str_skeleton` on the *string*
  form instead) and `create_dataset_from_raw_eqn`/`evaluate_eqn_list_on_datum`
  (point-sampling directly from the op-tree, unused by `dataGen`'s actual
  return path -- point sampling here instead goes through
  `symbolicgpt.utils.sample_points_for_equation`, which evaluates the
  string form the same way `CharDataset`'s on-the-fly augmentation does).
- Dropped the unused `wrapt_timeout_decorator` import (it only guarded a
  commented-out `@timeout(5)` decorator on `dataGen` in the original).
"""

import re

import numpy as np
import sympy
from sympy import sympify, expand

eps = 1e-4
big_eps = 1e-3

_DEFAULT_OP_LIST = ["id", "add", "sub", "mul", "div", "sin", "cos", "pow"]


def safe_abs(x):
    return np.sqrt(x * x + eps)


def safe_div(x, y):
    return np.sign(y) * x / safe_abs(y)


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def generate_random_eqn_raw(n_levels=2, n_vars=2, op_list=_DEFAULT_OP_LIST,
                            allow_constants=True, const_range=(-0.4, 0.4),
                            const_ratio=0.8, exponents=(3, 4, 5, 6)):
    """
    Random binary-tree equation as (ops, vars, weights, biases, exponents)
    lists. Variables are numbered 1..n_vars (0 never appears). `const_ratio`
    controls what fraction of weights/biases are non-trivial (i.e. not 1/0).
    """
    level_to_use = np.random.randint(1, n_levels)
    const_min_val, const_max_val = const_range
    eqn_ops = list(np.random.choice(op_list, size=int(2**level_to_use) - 1, replace=True))
    eqn_vars = list(np.random.choice(range(1, (n_vars + 1)), size=int(2 ** level_to_use), replace=True))
    max_bound = max(np.abs(const_min_val), np.abs(const_max_val))
    eqn_weights = list(np.random.uniform(-1 * max_bound, max_bound, size=len(eqn_vars)))
    eqn_biases = list(np.random.uniform(-1 * max_bound, max_bound, size=len(eqn_vars)))
    exponent_list = [exponents[np.random.randint(len(exponents))] for _ in range(2 ** level_to_use)]

    if not allow_constants:
        const_ratio = 0.0
    random_const_chooser_w = np.random.uniform(0, 1, len(eqn_weights))
    random_const_chooser_b = np.random.uniform(0, 1, len(eqn_biases))

    for i in range(len(eqn_weights)):
        if random_const_chooser_w[i] >= const_ratio:
            eqn_weights[i] = 1
        if random_const_chooser_b[i] >= const_ratio:
            eqn_biases[i] = 0

    return [eqn_ops, eqn_vars, eqn_weights, eqn_biases, exponent_list]


def raw_eqn_to_str(raw_eqn, n_vars=2):
    eqn_ops = raw_eqn[0]
    eqn_vars = raw_eqn[1]
    eqn_weights = raw_eqn[2]
    eqn_biases = raw_eqn[3]
    exponent_list = raw_eqn[4]
    current_op = eqn_ops[0]

    exponent = exponent_list[0]
    if len(eqn_ops) == 1:
        left_side = "({}*x{}+{})".format(float(eqn_weights[0]), eqn_vars[0], float(eqn_biases[0]))
        right_side = "({}*x{}+{})".format(float(eqn_weights[1]), eqn_vars[1], float(eqn_biases[1]))
    else:
        split_point = int((len(eqn_ops) + 1) / 2)
        left_ops = eqn_ops[1:split_point]
        right_ops = eqn_ops[split_point:]

        left_vars = eqn_vars[:split_point]
        right_vars = eqn_vars[split_point:]

        left_weights = eqn_weights[:split_point]
        left_biases = eqn_biases[:split_point]

        right_weights = eqn_weights[split_point:]
        right_biases = eqn_biases[split_point:]

        left_exponent = exponent_list[:split_point]
        right_exponent = exponent_list[split_point:]

        left_side = raw_eqn_to_str([left_ops, left_vars, left_weights, left_biases, left_exponent], n_vars)
        right_side = raw_eqn_to_str([right_ops, right_vars, right_weights, right_biases, right_exponent], n_vars)

    left_is_float = is_float(left_side)
    right_is_float = is_float(right_side)
    left_value = float(left_side) if left_is_float else np.nan
    right_value = float(right_side) if right_is_float else np.nan

    if current_op == 'id':
        return left_side
    if current_op == 'sqrt':
        return "{:.3f}".format(np.sqrt(np.abs(left_value))) if left_is_float else "sqrt(abs({}))".format(left_side)
    if current_op == 'log':
        return "{:.3f}".format(np.log(safe_abs(left_value))) if left_is_float else "log({})".format(left_side)
    if current_op == 'sin':
        return "{:.3f}".format(np.sin(left_value)) if left_is_float else "sin({})".format(left_side)
    if current_op == 'pow':
        return "{:.3f}".format(np.power(left_value, exponent)) if left_is_float else "({}**{})".format(left_side, exponent)
    if current_op == 'cos':
        return "{:.3f}".format(np.cos(left_value)) if left_is_float else "cos({})".format(left_side)
    if current_op == 'exp':
        return "{:.3f}".format(np.exp(left_value)) if left_is_float else "exp({})".format(left_side)
    if current_op == 'add':
        return "{:.3f}".format(left_value + right_value) if (left_is_float and right_is_float) else "({}+{})".format(left_side, right_side)
    if current_op == 'mul':
        return "{:.3f}".format(left_value * right_value) if (left_is_float and right_is_float) else "({}*{})".format(left_side, right_side)
    if current_op == 'sub':
        return "{:.3f}".format(left_value - right_value) if (left_is_float and right_is_float) else "({}-{})".format(left_side, right_side)
    if current_op == 'div':
        return "{:.3f}".format(safe_div(left_value, right_value)) if (left_is_float and right_is_float) else "({}/{})".format(left_side, right_side)
    return None


def simplify_formula(formula_to_simplify, digits=4):
    orig_form_str = sympify(formula_to_simplify)
    orig_form_str = expand(orig_form_str)
    rounded = orig_form_str

    try:
        for a in sympy.preorder_traversal(orig_form_str):
            if isinstance(a, sympy.Float):
                if np.abs(a) < 10**(-1 * digits):
                    rounded = rounded.subs(a, 0)
                else:
                    rounded = rounded.subs(a, round(a, digits))
    except Exception:
        return None

    return "{}".format(rounded).replace(' ', '')


def eqn_to_str(raw_eqn, n_vars=2, decimals=2):
    return simplify_formula(raw_eqn_to_str(raw_eqn, n_vars), digits=decimals)


def eqn_to_str_skeleton(eq):
    """Abstract a concrete equation string into a `C`-templated skeleton by
    regex substitution -- operates on the string form (not the op-tree),
    matching the original algorithm exactly."""
    eq = re.sub(r"\d+\.\d+", "C", eq)
    eq = eq.replace('-', '+')
    eq = re.sub(r'^\+', '', eq)

    eq = eq + '+C'  # add a bias
    dic = {
        '(+': '(',
        'sin(': 'sin(C*',
        'cos(': 'cos(C*',
        'log(': 'log(C*',
        'exp(': 'exp(C*',
        'sqrt(': 'sqrt(C*',
        'sin': 'C*sin',
        'cos': 'C*cos',
        'log': 'C*log',
        'exp': 'C*exp',
        'abs': 'C*abs',
        'sqrt': 'C*sqrt',
        'C**2': 'C', 'C**3': 'C', 'C**4': 'C', 'C**5': 'C',
        'C**6': 'C', 'C**7': 'C', 'C**8': 'C', 'C**9': 'C',
        'C*C*C': 'C',
        'C*C': 'C',
        'C+C': 'C',
    }
    for k in dic:
        eq = eq.replace(k, dic[k])
    return eq


def generate_equation(num_vars, decimals=4, n_levels=3, allow_constants=True,
                      const_range=(-0.4, 0.4), const_ratio=0.8,
                      op_list=_DEFAULT_OP_LIST, exponents=(2, 3)):
    """
    Generate one random equation over `num_vars` variables (x1..x_num_vars).

    Returns
    -------
    clean_eqn : str
        A concrete equation string (e.g. "1.234*sin(x1)+0.5").
    skeleton_eqn : str
        The same equation with numeric constants replaced by the
        placeholder token `C` (e.g. "C*sin(x1)+C").
    """
    raw_eqn = generate_random_eqn_raw(
        n_vars=num_vars, n_levels=n_levels, op_list=op_list,
        allow_constants=allow_constants, const_range=const_range,
        const_ratio=const_ratio, exponents=exponents,
    )
    clean_eqn = eqn_to_str(raw_eqn, n_vars=num_vars, decimals=decimals)
    skeleton_eqn = eqn_to_str_skeleton(clean_eqn)
    return clean_eqn, skeleton_eqn
