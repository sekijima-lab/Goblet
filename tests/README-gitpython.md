# GitPython security update validation

GitPython **3.1.43 → 3.1.62** addresses reviewed security advisories including
[CVE-2026-87817 / GHSA-239g-whfq-7xj9](https://github.com/advisories/GHSA-239g-whfq-7xj9).
Tracked files must not impersonate the real `.git` directory and its configuration.
This is defensive dependency maintenance, not evidence of exploitation of Goblet.

In a new Python 3.10 virtual environment:

```sh
python -m pip install -r tests/requirements-gitpython.txt
python tests/test_gitpython_compatibility.py
python -m pip check
```

The real WandB **0.18.7** Git client is checked for commit, branch, email, remote
URL, dirty/untracked state and diffs. Local clone, bare repository and linked
worktree operations are checked. These two compatibility tests pass before
and after, with the other **19** validation dependencies held identical and
matching the declared Goblet pins where specified.

A fixture uses only inert tracked files (`HEAD`, `objects`, `refs`, `gitdir`,
`commondir`, `config`): discovery and config isolation fail on 3.1.43 and pass
on 3.1.62. No executable hook or external config include is created.
All **3** updated checks and dependency checks pass on macOS arm64 / Python
3.10.19. OSV returned no advisories for 3.1.62 on 2026-10-02.

To compare the baseline, replace only the GitPython pin with 3.1.43 in a
second environment. `GITPYTHON_BASELINE=1` skips the expected failing
security check; run without it to reproduce the old discovery issue.

The previous **3 MLflow 3.16.1 tracking tests** also pass after updating only
GitPython in the isolated 91-package tracking validation environment. Its
frozen test requirements are updated to the same GitPython pin.

No model code or other runtime pins change. Full Linux/CUDA solving,
GPU training, molecular generation, server/network tracking and remaining
dependency advisories are outside this PR. These scoped test requirements
retain old dependencies and are not a general secure lockfile.
