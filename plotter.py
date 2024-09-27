import matplotlib.pyplot as plt
import numpy as np
import json
import os
def plot_evolution(fit_trackers, save=False, save_path="evolution_plot.png", show=False):
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

    if save:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()


def plot_box(individual_gains, save=False, save_path="boxplot.png", show=False, save_dir="mean_gains"):
    labels = list(individual_gains.keys())
    data = list(individual_gains.values())
    print(len(labelss))
    print(len(data))

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labelss)

    plt.title("Boxplot of Individual Gains by Algorithm/Enemy")
    plt.ylabel("Gains")

    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        full_save_path = os.path.join(save_dir, save_path)
        plt.savefig(full_save_path, bbox_inches="tight")
        print(f"Boxplot saved to {full_save_path}")

    if show:
        plt.show()

def save_individual_gains(individual_gains_by_enemy, algo_name, save_dir="mean_gains"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f"{algo_name}_mean_gains.json"
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'w') as json_file:
        json.dump(individual_gains_by_enemy, json_file)
    print(f"Individual gains saved to {file_name}")

def save_evolution_data(fit_trackers, algo_name, enemy, target_directory="evolution_data"):
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    file_name = f"{algo_name}_enemy{enemy}_evolution_data.json"
    file_name = f"{target_directory}/{file_name}"
    with open(file_name, 'w') as json_file:
        json.dump(fit_trackers, json_file)
    print(f"Evolution data saved to {file_name}")

def plot_evolution3x3(target_directory="evolution_data", save=True, save_path="evolution_plot.png", algo_names =["NEAT","GA","CMA-ES"]):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, algo_name in enumerate(algo_names):
        for j in range(3):
            with open(f"{target_directory}/{algo_name}_enemy{j+1}_evolution_data.json", 'r') as json_file:
                fit_trackers = json.load(json_file)
            X = range(1, len(fit_trackers[0]['mean'])+1)
            means = np.array([f["mean"] for f in fit_trackers])
            maxs = np.array([f["max"] for f in fit_trackers])
            mean_mean = np.mean(means, axis=0)
            std_mean = np.std(means, axis=0)
            mean_max = np.mean(maxs, axis=0)
            std_max = np.std(maxs, axis=0)
            axs[i, j].plot(X, mean_mean, label="Mean fitness", color="blue")
            axs[i, j].fill_between(X, mean_mean-std_mean, mean_mean+std_mean, color="blue", alpha=0.3)
            axs[i, j].fill_between(X, mean_mean-2*std_mean, mean_mean+2*std_mean, color="blue", alpha=0.2)
            axs[i, j].plot(X, mean_max, label="Max fitness", color="red")
            axs[i, j].fill_between(X, mean_max-std_max, mean_max+std_max, color="red", alpha=0.3)
            axs[i, j].fill_between(X, mean_max-2*std_max, mean_max+2*std_max, color="red", alpha=0.2)
            if i==0: #only plot for first row
                axs[i, j].set_title(f"Enemy {j+1}")
            if j==0: #only plot for first column
                axs[i, j].set_ylabel(algo_name)    
    fig.subylabel("Fitness")
    fig.sup_xlabel("Generation")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    if save:
        save_path = f"{target_directory}/{save_path}"
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()