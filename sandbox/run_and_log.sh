#!/bin/bash

# 1. Set parameter file
DEFAULT_CONFIG_FILE="params/default.yaml"
CONFIG_FILE=$DEFAULT_CONFIG_FILE
PYTHON_ARGS=()

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config_name)
        CONFIG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        --debug)
        PYTHON_ARGS+=("--debug")
        shift # past argument
        ;;
        *)    # pass to python script
        PYTHON_ARGS+=("$1")
        shift
        ;;
    esac
done

# Check if the specified config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at: $CONFIG_FILE"
    exit 1
fi

export PARAMS_FILE=$CONFIG_FILE
echo "Using parameter file: ${PARAMS_FILE}"

# 2. Generate nichika_run_id and create current_run.yaml
timestamp=$(TZ=Asia/Tokyo date +%Y%m%d%H%M%S)
nichika_run_id="exp_${timestamp}"
echo "nichika_run_id: ${nichika_run_id}" > current_run.yaml

# 3. Run the training
echo "Starting experiment ${nichika_run_id}..."

# Pass the collected arguments to the python script
poetry run python main.py "${PYTHON_ARGS[@]}"

# 4. The result saving is handled by the python script.
# The python script is modified to read current_run.yaml and save the results
# to ./result/training_[nichika_run_id]

echo "Experiment ${nichika_run_id} finished."
