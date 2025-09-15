import os
import json
import csv
import glob
from pathlib import Path

def make_dashboard():
    results_dir = 'result'
    output_dir = 'dashboard'
    output_file = os.path.join(output_dir, 'summary.csv')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find all experiment directories
    experiment_dirs = glob.glob(os.path.join(results_dir, 'training_exp_*'))

    all_results = []
    all_fieldnames = set()

    for exp_dir in experiment_dirs:
        config_path = os.path.join(exp_dir, 'config.json')
        metrics_path = os.path.join(exp_dir, 'logs', 'test_metrics.json')

        if not os.path.exists(config_path) or not os.path.exists(metrics_path):
            continue

        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)

        # Combine config and metrics
        exp_name = os.path.basename(exp_dir)
        combined_data = {'experiment_name': exp_name, **config_data, **metrics_data}
        all_results.append(combined_data)
        all_fieldnames.update(combined_data.keys())

    if not all_results:
        print("No results found to aggregate.")
        return

    # Write to CSV
    sorted_fieldnames = sorted(list(all_fieldnames))
    if 'experiment_name' in sorted_fieldnames:
        sorted_fieldnames.remove('experiment_name')
        sorted_fieldnames.insert(0, 'experiment_name')
        
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Dashboard created at: {output_file}")

if __name__ == '__main__':
    make_dashboard()
