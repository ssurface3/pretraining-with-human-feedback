#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ $# -eq 0 ]]; then
  set -- --task configs/toxicity/pretrain.yml --method configs/toxicity/conditional.yml
fi

if ! command -v python3.13 >/dev/null 2>&1; then
  echo "python3.13 is required" >&2
  exit 1
fi

python3.13 -m venv --clear .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export COMET_MODE="${COMET_MODE:-offline}"

python train.py "$@"
