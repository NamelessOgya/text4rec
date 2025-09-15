
import os
import json
import pandas as pd

def analyze_results(experiment_description):
    """Analyzes results from all experiments with the given description."""
    results = []
    experiment_dir = 'experiments'

    print(f"Searching for experiments with description starting with: {experiment_description}")

    if not os.path.exists(experiment_dir):
        print(f"Error: Experiment directory '{experiment_dir}' not found.")
        return

    for dirname in sorted(os.listdir(experiment_dir)):
        if dirname.startswith(experiment_description):
            print(f"Found experiment directory: {dirname}")
            config_path = os.path.join(experiment_dir, dirname, 'config.json')
            metrics_path = os.path.join(experiment_dir, dirname, 'logs', 'test_metrics.json')

            if os.path.exists(config_path) and os.path.exists(metrics_path):
                print(f"  - Reading config from: {config_path}")
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                print(f"  - Reading metrics from: {metrics_path}")
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)

                # Combine config and metrics into a single dictionary for this run
                run_result = {
                    'experiment_name': dirname, 
                    **config,
                    **metrics
                }
                results.append(run_result)
            else:
                print(f"  - Warning: config.json or test_metrics.json not found in {dirname}")

    if not results:
        print("No results found for the given description.")
        return

    # Create a pandas DataFrame for easy analysis
    df = pd.DataFrame(results)

    # Select and reorder columns for clarity
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    
    # Example: Displaying key parameters and metrics
    key_columns = [
        'experiment_name',
        'lr',
        'bert_dropout',
        'num_epochs',
        'NDCG@10',
        'Recall@10'
    ]
    # Filter for columns that exist in the DataFrame
    display_columns = [col for col in key_columns if col in df.columns]

    print("\n--- Experiment Results Summary ---")
    print(df[display_columns])

    # Save all results to a CSV file
    output_path = f'sandbox/results_summary_{experiment_description}.csv'
    df.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

if __name__ == '__main__':
    # Specify the experiment description you want to analyze
    # This should match the description used in your experiment script
    description_to_analyze = "bertemb_multiple_runs"
    analyze_results(description_to_analyze)
