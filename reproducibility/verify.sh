#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

sha256sum --check reproducibility/checksums.sha256
bash -n 0_generate.job.sh batch.sh reproducibility/verify.sh
python -c 'from pathlib import Path; [compile(Path(name).read_text(), name, "exec") for name in ("Generator/mcts.py", "Utils/boltz_api_constraints.py", "config/config.py")]'

echo "Reproducibility assets verified."
