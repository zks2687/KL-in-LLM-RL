#!/usr/bin/env bash
# Shared by the training launchers. Source this file; do not put API keys here.

PYTHON_BIN=${PYTHON_BIN:-python3}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}

# Honor the final Hydra logger override before checking optional dependencies.
for kl_logging_arg in "$@"; do
    if [[ "$kl_logging_arg" == trainer.logger=* ]]; then
        TRAINER_LOGGER=${kl_logging_arg#trainer.logger=}
    fi
done

if [[ "$TRAINER_LOGGER" == *swanlab* ]]; then
    if ! "$PYTHON_BIN" -c 'import swanlab' >/dev/null 2>&1; then
        printf '%s\n' 'SwanLab is unavailable in the active Python environment.' \
            "Install it with: $PYTHON_BIN -m pip install swanlab" \
            'Then run: swanlab login (or provide SWANLAB_API_KEY through your environment).' >&2
        exit 1
    fi

    export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
    export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/swanlog"}
    mkdir -p -- "$SWANLAB_LOG_DIR"

    # Opt in to interactive login before starting Ray; otherwise reuse saved
    # credentials or SWANLAB_API_KEY. No credentials are printed by this script.
    if [[ ${SWANLAB_LOGIN:-0} == 1 && "$SWANLAB_MODE" == cloud ]]; then
        swanlab login
    fi
fi
