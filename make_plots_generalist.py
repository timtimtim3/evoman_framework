import json
import os
import argparse
from plotter import plot_box, plot_evolution_groups
from scipy.stats import ttest_ind
import pandas as pd
import ast  # For safely evaluating string representations of lists
import numpy as np

def load_and_combine_gains(algo_names, save_dir="mean_gains_generalist", train_group_1=None, train_group_2=None):
    combined_gains = {}

    for algo_name in algo_names:
        file_name = f"{algo_name}_mean_gains.json"
        file_path = os.path.join(save_dir, file_name)

        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                gains = json.load(json_file)
                combined_gains.update(gains)
            print(f"Loaded gains from {file_path}")
        else:
            print(f"File {file_path} not found")

    print("Original combined gains:", combined_gains)

    # Adjust keys to replace train group list with 'A' or 'B'
    adjusted_gains = {}
    for key, value in combined_gains.items():
        # key is like 'CMA-ES [1, 2, 3, 4, 6, 7, 8]'
        # Extract algorithm name and train group list
        algorithm_name, train_group_str = key.split(' ', 1)
        # Safely evaluate train group list from string
        train_group_list = ast.literal_eval(train_group_str)
        # Compare train_group_list with train_group_1 and train_group_2
        if sorted(train_group_list) == sorted(train_group_1):
            group_label = 'A'
        elif sorted(train_group_list) == sorted(train_group_2):
            group_label = 'B'
        else:
            group_label = 'Unknown'

        new_key = f"{algorithm_name} {group_label}"
        adjusted_gains[new_key] = value

    # Now sort adjusted_gains
    sorted_adjusted_gains = dict(sorted(
        adjusted_gains.items(),
        key=lambda x: (x[0].split()[1], x[0].split()[0])
    ))

    print("Adjusted and sorted gains:", sorted_adjusted_gains)
    return sorted_adjusted_gains


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load gains from multiple algorithms")

    parser.add_argument('--algo_names', type=str, nargs='+', default=['CMA-ES', 'GA'],
                        help="List of algorithm names to load gains from.")
    parser.add_argument('--mean_gains_save_dir', type=str, default="mean_gains_generalist",
                        help="Directory from which to load the mean gains.")
    parser.add_argument('--evolution_data_dir', type=str, default="evolution_data_generalist",
                        help="Directory from which to load the evolution data.")
    parser.add_argument("--train_group_1", type=int, nargs='+', default=[1, 2, 3, 4, 6, 7, 8],
                        help="First training group (will be labeled as 'A').")
    parser.add_argument("--train_group_2", type=int, nargs='+', default=[1, 2, 3],
                        help="Second training group (will be labeled as 'B').")

    args = parser.parse_args()

    combined_gains = load_and_combine_gains(
        args.algo_names,
        save_dir=args.mean_gains_save_dir,
        train_group_1=args.train_group_1,
        train_group_2=args.train_group_2
    )

    plot_box(combined_gains, save=True, save_path="combined_boxplot.png", save_dir=args.mean_gains_save_dir,
             rotation=45)

    plot_evolution_groups(
        args.algo_names,
        train_group_1=args.train_group_1,
        train_group_2=args.train_group_2,
        save=True,
        show=True,
        save_dir=args.evolution_data_dir,
        plot_spreads=True
    )

    # Perform t-tests between CMA-ES and GA for each group
    group_results = {}

    for group_label in ['A', 'B']:
        cmaes_key = f'CMA-ES {group_label}'
        ga_key = f'GA {group_label}'

        if cmaes_key in combined_gains and ga_key in combined_gains:
            cmaes_gains = combined_gains[cmaes_key]
            ga_gains = combined_gains[ga_key]

            # Calculate means and standard errors
            cmaes_mean = np.mean(cmaes_gains)
            ga_mean = np.mean(ga_gains)
            cmaes_std = np.std(cmaes_gains, ddof=1)  # Sample standard deviation
            ga_std = np.std(ga_gains, ddof=1)  # Sample standard deviation
            cmaes_sem = cmaes_std / np.sqrt(len(cmaes_gains))
            ga_sem = ga_std / np.sqrt(len(ga_gains))

            # Perform t-test
            t_stat, p_value = ttest_ind(cmaes_gains, ga_gains, equal_var=False)

            # Store the results
            group_results[group_label] = {
                'CMA-ES Gains': cmaes_gains,
                'GA Gains': ga_gains,
                'CMA-ES Mean': cmaes_mean,
                'GA Mean': ga_mean,
                'CMA-ES SEM': cmaes_sem,
                'GA SEM': ga_sem,
                't-statistic': t_stat,
                'p-value': p_value
            }

            print(f"\nGroup {group_label} T-test between CMA-ES and GA:")
            print(f"CMA-ES Mean Gain: {cmaes_mean:.4f} ± {cmaes_sem:.4f} (SEM)")
            print(f"GA Mean Gain: {ga_mean:.4f} ± {ga_sem:.4f} (SEM)")
            print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
        else:
            print(f"\nData for group {group_label} is incomplete. Cannot perform t-test.")

    # Save the t-test results to a CSV file
    ttest_results = []
    for group_label, results in group_results.items():
        ttest_results.append({
            'Group': group_label,
            'CMA-ES Mean Gain': results['CMA-ES Mean'],
            'CMA-ES SEM': results['CMA-ES SEM'],
            'GA Mean Gain': results['GA Mean'],
            'GA SEM': results['GA SEM'],
            't-statistic': results['t-statistic'],
            'p-value': results['p-value']
        })

    df = pd.DataFrame(ttest_results)
    results_file = os.path.join(args.mean_gains_save_dir, 'ttest_results.csv')
    df.to_csv(results_file, index=False)
    print(f"\nT-test results saved to {results_file}")