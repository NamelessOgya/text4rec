#!/bin/bash

# Comprehensive experiment script for comparing quantization methods and ablation studies
# Each condition is run 3 times with different seeds.

SEEDS=(0820 0831 0924)
# SEEDS=(0831)
LR=0.001
NEG_SIZE=64
description_prefix="11021439"

# --- No Quantization (Base Model) ---
echo "--- Experiments: SASRec (No Quantization) - Base ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_random.yaml --experiment_description "${description_prefix}_sasrec_no_quant_base_${seed}" --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done

echo "--- Experiments: SASRec (No Quantization) - Semi-synthetic NS ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_random.yaml --experiment_description "${description_prefix}_sasrec_no_quant_semi_ns_${seed}" --use_semi_synthetic_ns --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done


#--- Static Quantization ---
echo "--- Experiments: SASRec (Static Quantization) - Base ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_quantize.yaml --experiment_description "${description_prefix}_sasrec_static_quant_base_${seed}" --quantizer_type "static" --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done

echo "--- Experiments: SASRec (Static Quantization) - Semi-synthetic NS ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_quantize.yaml --experiment_description "${description_prefix}_sasrec_static_quant_semi_ns_${seed}" --quantizer_type "static" --use_semi_synthetic_ns --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done

echo "--- Experiments: SASRec (Static Quantization) - Code Alignment ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_quantize.yaml --experiment_description "${description_prefix}_sasrec_static_quant_align_${seed}" --quantizer_type "static" --use_code_alignment_loss --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done

echo "--- Experiments: SASRec (Static Quantization) - Semi-synthetic NS + Code Alignment ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_quantize.yaml --experiment_description "${description_prefix}_sasrec_static_quant_semi_ns_align_${seed}" --quantizer_type "static" --use_semi_synthetic_ns --use_code_alignment_loss --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done


# --- Dynamic Quantization ---
echo "--- Experiments: SASRec (Dynamic Quantization) - Base ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_quantize.yaml --experiment_description "${description_prefix}_sasrec_dynamic_quant_base_${seed}" --quantizer_type "dynamic" --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done

echo "--- Experiments: SASRec (Dynamic Quantization) - Semi-synthetic NS ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_quantize.yaml --experiment_description "${description_prefix}_sasrec_dynamic_quant_semi_ns_${seed}" --quantizer_type "dynamic" --use_semi_synthetic_ns --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done

echo "--- Experiments: SASRec (Dynamic Quantization) - Code Alignment ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_quantize.yaml --experiment_description "${description_prefix}_sasrec_dynamic_quant_align_${seed}" --quantizer_type "dynamic" --use_code_alignment_loss --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done

echo "--- Experiments: SASRec (Dynamic Quantization) - Semi-synthetic NS + Code Alignment ---"
for seed in "${SEEDS[@]}"; do
    ./sandbox/run_and_log.sh --config_name params/sasrec_quantize.yaml --experiment_description "${description_prefix}_sasrec_dynamic_quant_semi_ns_align_${seed}" --quantizer_type "dynamic" --use_semi_synthetic_ns --use_code_alignment_loss --lr ${LR} --train_negative_sample_size ${NEG_SIZE} --model_init_seed ${seed} --train_negative_sampling_seed ${seed}
done

