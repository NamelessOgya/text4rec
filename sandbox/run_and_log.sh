#!/bin/bash

# 1. Set parameter file
DEFAULT_CONFIG_FILE="params/default.yaml"
CONFIG_FILE=$DEFAULT_CONFIG_FILE

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config_name)
        CONFIG_FILE="$2"
        shift # past argument
        shift # past value
        ;;
        *)    # unknown option
        echo "Unknown option for run_and_log.sh: $1"
        exit 1
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
timestamp=$(date +%Y%m%d%H%M%S)
nichika_run_id="exp_${timestamp}"
echo "nichika_run_id: ${nichika_run_id}" > current_run.yaml

# 3. Run emb_b4r.sh
echo "Starting experiment ${nichika_run_id}..."

# Note: emb_b4r.sh has an interactive prompt. 
# To run this script in a fully automated way, you might need to modify emb_b4r.sh 
# or pipe `yes` to it, like: yes | ./test/emb_b4r.sh
./test/emb_b4r.sh

# 4. The result saving is handled by the python script.
# The python script is modified to read current_run.yaml and save the results
# to ./result/training_[nichika_run_id]

echo "Experiment ${nichika_run_id} finished."
