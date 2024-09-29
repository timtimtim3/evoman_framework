import json
import os
import argparse
from plotter import plot_box, plot_evolution3x3


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load gains from multiple algorithms")

    parser.add_argument('--algo_names', type=str, nargs='+', default=['CMA-ES', 'NEAT', 'GA'],
                        help="List of algorithm names to load gains from.")
    parser.add_argument('--mean_gains_save_dir', type=str, default="mean_gains",
                        help="Directory from which to load the mean gains.")

    args = parser.parse_args()

    combined_gains = load_and_combine_gains(args.algo_names, save_dir=args.mean_gains_save_dir)

    print(combined_gains)
    plot_box(combined_gains, save=True, save_path="combined_boxplot.png")

    plot_evolution3x3()
