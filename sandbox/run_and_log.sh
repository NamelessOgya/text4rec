#!/bin/bash

# 1. Set parameter file
PARAM_NAME=${1:-default}
export PARAMS_FILE="params/${PARAM_NAME}.yaml"
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
