#!/usr/bin/env bash

set -euo pipefail

GOBLET_ENV="${GOBLET_ENV:-goblet}"
BOLTZ_ENV="${BOLTZ_ENV:-$GOBLET_ENV}"
qsub_options=()
if [[ -n "${GROUP:-}" ]]; then
    qsub_options+=(-g "$GROUP")
fi
if [[ -n "${MAIL:-}" ]]; then
    qsub_options+=(-M "$MAIL" -m eas)
fi

for target in akt1 ampc cxcr4 cp3a4 hivrt hivpr gcr kif11; do
    qsub "${qsub_options[@]}" \
        -N "$target" \
        -v "TARGET=${target},GOBLET_ENV=${GOBLET_ENV},BOLTZ_ENV=${BOLTZ_ENV}" \
        0_generate.job.sh
done
