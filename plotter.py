import matplotlib.pyplot as plt
import numpy as np
import json



def plot_evolution(fit_trackers, save=False, save_path="evolution_plot.png"):
    X = range(1, len(fit_trackers[0]['mean'])+1)
    means = np.array([f["mean"] for f in fit_trackers])
    maxs = np.array([f["max"] for f in fit_trackers])
    mean_mean = np.mean(means, axis=0)
    std_mean = np.std(means, axis=0)
    mean_max = np.mean(maxs, axis=0)
    std_max = np.std(maxs, axis=0)
    plt.plot(X, mean_mean, label="Mean fitness", color="blue")
    plt.fill_between(X, mean_mean-std_mean, mean_mean+std_mean, color="blue", alpha=0.3)
    plt.fill_between(X, mean_mean-2*std_mean, mean_mean+2*std_mean, color="blue", alpha=0.2)
    plt.plot(X, mean_max, label="Max fitness", color="red")
    plt.fill_between(X, mean_max-std_max, mean_max+std_max, color="red", alpha=0.3)
    plt.fill_between(X, mean_max-2*std_max, mean_max+2*std_max, color="red", alpha=0.2)
    plt.title("Fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_box(individual_gains, save=False, save_path="boxplot.png"):
    labelss = list(individual_gains.keys())
    data = list(individual_gains.values())
    print(len(labelss))
    print(len(data))

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labelss)

    plt.title("Boxplot of Individual Gains by Algorithm/Enemy")
    # plt.xlabel("Algorithm & Enemy")
    plt.ylabel("Gains")

    plt.xticks(rotation=90, ha="right")

    if save:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Boxplot saved to {save_path}")

    plt.tight_layout()
    plt.show()

def save_individual_gains(individual_gains_by_enemy, algo_name):
    file_name = f"{algo_name}_mean_gains.json"
    with open(file_name, 'w') as json_file:
        json.dump(individual_gains_by_enemy, json_file)
    print(f"Individual gains saved to {file_name}")
