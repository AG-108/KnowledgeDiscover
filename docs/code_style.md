# Code style

The project uses Black for Python layout and Ruff for import ordering and static style checks.
Both tools read their settings from `pyproject.toml`. Editor defaults such as UTF-8, LF line
endings, indentation, and trailing whitespace are defined in `.editorconfig`.

Install the development tools from the repository root:

```bash
python -m pip install -e ".[dev]"
```

Format maintained Python code and fix import ordering:

```bash
black run_benchmark.py kd examples tests docs/source/conf.py
ruff check --select I --fix run_benchmark.py kd examples tests docs/source/conf.py
```

Run the checks without changing files:

```bash
black --check run_benchmark.py kd examples tests docs/source/conf.py
ruff check run_benchmark.py kd examples tests docs/source/conf.py
bash -n scripts/run_benchmark.sh
```

Use concise English docstrings for public modules, classes, and functions. Write inline comments
as complete sentences that explain intent, constraints, or non-obvious behavior. Avoid comments
that merely repeat the following statement, numbered implementation steps, dead commented code,
and wildcard imports.

The directories excluded in `pyproject.toml` are embedded upstream projects and archived source
snapshots. Keep their original formatting so that local changes remain easy to compare and sync
with their upstream repositories.
