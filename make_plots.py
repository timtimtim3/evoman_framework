import json
import os
import argparse
from plotter import plot_box, plot_evolution3x3
from scipy.stats import ttest_ind
import pandas as pd


def load_and_combine_gains(algo_names, save_dir="mean_gains"):
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

    sorted_combined_gains = dict(sorted(combined_gains.items(), key=lambda x: (x[0].split()[0], int(x[0].split()[-1]))))

    return sorted_combined_gains


def perform_t_tests(combined_gains):
    algo_names = list(set(key.split()[0] for key in combined_gains.keys()))
    enemies = list(set(int(key.split()[-1]) for key in combined_gains.keys()))
    t_test_results = {}

    def join(algo, enemy):
        return f"{algo} {enemy}"
    
    for enemy in enemies:
        enemy_results = {}
        for i in range(len(algo_names)):
            for j in range(i + 1, len(algo_names)):
                algo1, algo2 = algo_names[i], algo_names[j]
                gains1 = combined_gains[join(algo1, enemy)]
                gains2 = combined_gains[join(algo2, enemy)]
                if gains1 and gains2:
                    t_stat, p_value = ttest_ind(gains1, gains2)
                    enemy_results[f"{algo1} vs {algo2}"] = {"t_stat": t_stat, "p_value": p_value}

        t_test_results[f"Enemy {enemy}"] = enemy_results

    return t_test_results

def save_t_test_results_to_excel(t_test_results, file_name="t_test_results.xlsx"):
    # Create a list to hold the data for the DataFrame
    data = []

    # Iterate over the t_test_results dictionary to populate the data list
    for enemy, results in t_test_results.items():
        for comparison, stats in results.items():
            row = {
                "Enemy": enemy,
                "Comparison": comparison,
                "t_stat": stats["t_stat"],
                "p_value": stats["p_value"]
            }
            data.append(row)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel(file_name, index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load gains from multiple algorithms")

    parser.add_argument('--algo_names', type=str, nargs='+', default=['CMA-ES', 'NEAT', 'GA'],
                        help="List of algorithm names to load gains from.")
    parser.add_argument('--mean_gains_save_dir', type=str, default="mean_gains",
                        help="Directory from which to load the mean gains.")

    args = parser.parse_args()

    combined_gains = load_and_combine_gains(args.algo_names, save_dir=args.mean_gains_save_dir)

    
    t_test_results = perform_t_tests(combined_gains)
    save_t_test_results_to_excel(t_test_results)
    plot_box(combined_gains, save=True, save_path="combined_boxplot.png")
    plot_evolution3x3()
