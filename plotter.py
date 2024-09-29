import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_evolution(fit_trackers, save=False, save_path="evolution_plot.png", show=False, save_dir="evolution_data"):
    # Plots the evolution of a single agent-enemy pair
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
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        full_save_path = os.path.join(save_dir, save_path)
        plt.savefig(full_save_path)
        print(f"Plot saved to {full_save_path}")

    if show:
        plt.show()

    plt.close()


def plot_box(individual_gains, save=False, save_path="boxplot.png", show=False, save_dir="mean_gains"):
    # Plots a boxplot of the individual gains for each algorithm/enemy pair
    labels = list(individual_gains.keys())
    data = list(individual_gains.values())

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)

    # plt.title("Boxplot of Individual Gains by Algorithm/Enemy")
    plt.ylabel("Gains", fontsize =16)

    plt.xticks(rotation=90, ha="right", fontsize =12)
    plt.yticks(fontsize =12)
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
    # Saves the individual gains to a JSON file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = f"{algo_name}_mean_gains.json"
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'w') as json_file:
        json.dump(individual_gains_by_enemy, json_file)
    print(f"Individual gains saved to {file_name}")

def save_evolution_data(fit_trackers, algo_name, enemy, target_directory="evolution_data"):
    # Saves the evolution data to a JSON file
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    file_name = f"{algo_name}_enemy{enemy}_evolution_data.json"
    file_name = f"{target_directory}/{file_name}"
    with open(file_name, 'w') as json_file:
        json.dump(fit_trackers, json_file)
    print(f"Evolution data saved to {file_name}")

def plot_evolution3x3(target_directory="evolution_data", save=True, save_path="evolution_plot.png", algo_names =["NEAT","GA","CMA-ES"]):
    # Plots the evolution of all algorithms for all enemies in a 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharey=True, sharex=True)
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    for i, algo_name in enumerate(algo_names)  :
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
            axs[j, i].plot(X, mean_mean, label="Mean fitness", color="blue")
            axs[j, i].fill_between(X, mean_mean-std_mean, mean_mean+std_mean, color="blue", alpha=0.3, label = "Mean fitness std")	
            axs[j, i].fill_between(X, mean_mean-2*std_mean, mean_mean+2*std_mean, color="blue", alpha=0.2, label = "Mean fitness std*2")
            axs[j, i].plot(X, mean_max, label="Max fitness", color="red")
            axs[j, i].fill_between(X, mean_max-std_max, mean_max+std_max, color="red", alpha=0.3, label = "Max fitness std")
            axs[j, i].fill_between(X, mean_max-2*std_max, mean_max+2*std_max, color="red", alpha=0.2, label = "Max fitness std*2")
            
            if j==0: #only plot for first row
                axs[j, i].set_title(algo_name, fontsize =12)   
            if i==0: #only plot for first column
                axs[j, i].set_ylabel(f"Enemy {j+1}", fontsize =12) 
            
            axs[i,j].tick_params(labelleft=True, labelbottom=True)

    # X and Y labels    
    fig.text(0.04, 0.5, 'Fitness', va='center', rotation='vertical', fontsize=16)
    fig.text(0.5, 0.04, 'Generation', ha='center', fontsize=16)
    # set x ticks:
    for ax in axs.flat:
        ax.set_xticks([1, 10, 20, 30, 40, 50])
    

    handles, labels = axs[0, 0].get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), borderaxespad=0., fontsize =12)
    plt.subplots_adjust(right=0.763)  # Adjust the right side of the plot to make space for the legend
    if save:
        save_path = f"{target_directory}/{save_path}"
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()