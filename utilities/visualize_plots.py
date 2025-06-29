import argparse
import os
from utilities.utils import plot_multiple_results, export_metrics_to_json 

def analyze_experiment(shock_type, experiment_type, variable_name, 
                       cols=3, figsize_per_row=(15, 3)):

    logs_base_dir=f"logs/{shock_type}"
    pattern = os.path.join(logs_base_dir, experiment_type, variable_name, "*.csv")

    plot_multiple_results(pattern, cols=cols, figsize_per_row=figsize_per_row, save_dir=f"plots/{shock_type}/{experiment_type}", save_name=f"{variable_name}.png")

    export_metrics_to_json(pattern, output_dir=f"json_output/{shock_type}/{experiment_type}/{variable_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results by plotting and exporting metrics.")
    parser.add_argument("--experiment", type=str, required=True,
                        help="Name of the experiment folder (e.g., 'unsupervised_experiment')")
    parser.add_argument("--variable", type=str, required=True,
                        help="Name of the variable folder (e.g., 'lamb')")
    parser.add_argument("--shock_type", type=str, default='sweep', 
                        help="Do you want results when shock was performed?")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the plot grid")
    parser.add_argument("--row_height", type=float, default=3, help="Height of each row in the plot")

    args = parser.parse_args()

    analyze_experiment(args.shock_type, args.experiment, args.variable, cols=args.cols, figsize_per_row=(15, args.row_height))
