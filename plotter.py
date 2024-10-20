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


def plot_box(individual_gains, save=False, save_path="boxplot.png", show=False, save_dir="mean_gains", rotation=90):
    # Plots a boxplot of the individual gains for each algorithm/enemy pair
    labels = list(individual_gains.keys())
    data = list(individual_gains.values())

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)

    # plt.title("Boxplot of Individual Gains by Algorithm/Enemy")
    plt.ylabel("Gains", fontsize =16)

    plt.xticks(rotation=rotation, ha="right", fontsize =12)
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


def plot_evolution_groups(algo_names, train_group_1, train_group_2, save=False, save_path="evolution_plot.png",
                          show=False, save_dir="evolution_data_generalist", plot_spreads=True):
    """
    Plots the evolution of fitness over generations for different algorithms across two groups.

    Parameters:
    - algo_names (list): List of algorithm names (e.g., ['GA', 'CMA-ES']).
    - train_group_1 (list): Enemy list for group A.
    - train_group_2 (list): Enemy list for group B.
    - save (bool): Whether to save the plot as an image file.
    - save_path (str): Filename for the saved plot.
    - show (bool): Whether to display the plot.
    - save_dir (str): Directory where evolution data and plots are saved.
    - plot_spreads (bool): Whether to plot standard deviation spreads.
    """

    # Mapping group labels to their respective enemy lists
    groups = {'A': train_group_1, 'B': train_group_2}
    data = {}

    # Load evolution data for each group and algorithm
    for group_label, group_list in groups.items():
        data[group_label] = {}
        group_list_str = str(group_list)  # Converts list to string representation

        for algo_name in algo_names:
            # All files have '_evolution_data.json' as the extension
            file_extension = '_evolution_data.json'

            # Construct the file name
            file_name = f"{algo_name}_enemy{group_list_str}{file_extension}"
            file_path = os.path.join(save_dir, file_name)

            # Load the evolution data if the file exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as json_file:
                    fit_trackers = json.load(json_file)
                    data[group_label][algo_name] = fit_trackers
                print(f"Loaded evolution data from {file_path}")
            else:
                print(f"File {file_path} not found")
                data[group_label][algo_name] = None

    # Create a figure with two subplots side by side for groups A and B
    # Set sharey=False to allow independent y-axes
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), sharey=False)

    # Define distinct colors for each metric of each algorithm
    # Ensure that all four colors are clearly distinguishable
    color_mapping = {
        'GA_mean': '#1f77b4',  # Dark Blue
        'GA_max': '#005f9e',  # Vibrant Blue
        'CMA-ES_mean': '#d62728',  # Dark Red
        'CMA-ES_max': '#a50f15'  # Vibrant Red
    }

    for idx, (group_label, group_data) in enumerate(data.items()):
        ax = axs[idx]
        for algo_name, fit_trackers in group_data.items():
            if fit_trackers is None:
                continue

            # Extract mean and max fitness over generations
            X = range(1, len(fit_trackers[0]['mean']) + 1)
            means = np.array([f["mean"] for f in fit_trackers])
            maxs = np.array([f["max"] for f in fit_trackers])
            mean_mean = np.mean(means, axis=0)
            std_mean = np.std(means, axis=0)
            mean_max = np.mean(maxs, axis=0)
            std_max = np.std(maxs, axis=0)

            # Determine colors based on algorithm and metric
            color_mean = color_mapping.get(f"{algo_name}_mean", '#000000')  # Default to black
            color_max = color_mapping.get(f"{algo_name}_max", '#555555')  # Default to dark gray

            # Plot mean fitness with a solid line
            ax.plot(X, mean_mean, label=f"{algo_name} Mean Fitness",
                    color=color_mean, linestyle='-', linewidth=2)

            if plot_spreads:
                ax.fill_between(X, mean_mean - std_mean, mean_mean + std_mean,
                                color=color_mean, alpha=0.3)
                # Uncomment the following lines if you want to include double standard deviation
                # ax.fill_between(X, mean_mean - 2 * std_mean, mean_mean + 2 * std_mean,
                #                 color=color_mean, alpha=0.2)

            # Plot max fitness with a solid line
            ax.plot(X, mean_max, label=f"{algo_name} Max Fitness",
                    color=color_max, linestyle='-', linewidth=2)

            if plot_spreads:
                ax.fill_between(X, mean_max - std_max, mean_max + std_max,
                                color=color_max, alpha=0.3)
                # Uncomment the following lines if you want to include double standard deviation
                # ax.fill_between(X, mean_max - 2 * std_max, mean_max + 2 * std_max,
                #                 color=color_max, alpha=0.2)

        # Set titles and labels
        ax.set_title(f"Group {group_label}", fontsize=18)
        ax.set_xlabel("Generation", fontsize=14)
        ax.set_ylabel("Fitness", fontsize=14)

        # Add legend only to the right-hand subplot (index 1)
        if idx == 1:
            legend = ax.legend(fontsize=16, title="Metrics", title_fontsize=18, loc='best')
            legend.get_frame().set_edgecolor('black')  # Add a border to the legend
            legend.get_frame().set_linewidth(1.5)  # Set the border width

        # Add grid lines
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save the plot if required
    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        full_save_path = os.path.join(save_dir, save_path)
        plt.savefig(full_save_path, dpi=300)
        print(f"Plot saved to {full_save_path}")

    # Show the plot if required
    if show:
        plt.show()

    # Close the plot to free memory
    plt.close()




