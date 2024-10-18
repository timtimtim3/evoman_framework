import optuna
from optuna.trial import TrialState

import pymysql
pymysql.install_as_MySQLdb()
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import plotter
import json
import os



# study = optuna.create_study(study_name='run1overnight', direction="maximize", storage='mysql+pymysql://root:UbuntuMau@localhost/CMAES', load_if_exists=True)
study = optuna.create_study(study_name='run1', direction="maximize", storage='mysql+pymysql://root:UbuntuMau@localhost/GA', load_if_exists=True)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
best_trial = study.best_trial

print("  Value: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

for key, value in best_trial.user_attrs.items():
    print("    {}: {}".format(key, value))


os.makedirs('hypertune_params', exist_ok = True)
with open('hypertune_params/GA.json', 'w') as json_file:
    json.dump(best_trial.params, json_file, indent=3)

controller = np.array(best_trial.user_attrs['best_model_solution'])


env2 = Environment(experiment_name='test',
                      enemies=[1,2,3,4,5,6,7,8],
                      playermode="ai",
                      multiplemode="yes",
                      player_controller=player_controller(10),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=True)

env = Environment(experiment_name='test',
                      enemies=[1],
                      playermode="ai",
                    #   multiplemode="yes",
                      player_controller=player_controller(10),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=True)
enemies = [1,2,3,4,5,6,7,8]
means = []
win = []
for enemy in enemies:
    env.update_parameter('enemies',[enemy])
    f, p, e, t = env.play(pcont=controller)
    if p > 0:
        print("win")
        win.append(enemy)
    print(f, p, e, t)
    means.append(f)
print(f'fitness of {np.mean(means)} with {len(win)}/8 wins')
print(env2.play(pcont=controller))

from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_hypervolume_history,plot_pareto_front, plot_contour, plot_intermediate_values, plot_slice

params =['enemy 1', 'enemy 2', 'enemy 3', 'enemy 4', 'enemy 5', 'enemy 6', 'enemy 7', 'enemy 8']
params = ['sigma', 'Population size']
# print(optuna.importance.get_param_importances(study))
# plot_parallel_coordinate(study, params=params).show()
# plot_param_importances(study, params=params).show()
# plot_slice(study, params=params).show()
# optuna.visualization.plot_optimization_history(study).show()

# # plot_hypervolume_history(study)
# plot_contour(study, params=params).show()
# plot_intermediate_values(study).show()


# plotter.plot_evolution([best_trial.user_attrs['fit_tracker']],show=True)
