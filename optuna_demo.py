import optuna
import pymysql
pymysql.install_as_MySQLdb()
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import plotter
import json
import os



study = optuna.create_study(study_name='kiki', direction="maximize", storage='mysql+pymysql://root:UbuntuMau@localhost/test', load_if_exists=True)

best_trial = study.best_trial
print(best_trial.user_attrs["enemies"])

print("  Value: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

os.makedirs('hypertune_params', exist_ok = True)
with open('hypertune_params/CMA-ES-params.json', 'w') as json_file:
    json.dump(best_trial.params, json_file, indent=3)

controller = np.array(best_trial.user_attrs['best_model_solution'])


env2 = Environment(experiment_name='test',
                      enemies=[1,2,3,4,5,6,7,8],
                      playermode="ai",
                      multiplemode="yes",
                      player_controller=player_controller(10),
                      enemymode="static",
                      level=2,
                      speed="normal",
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
for enemy in enemies:
    env.update_parameter('enemies',[enemy])
    print(env.play(pcont=controller))
from optuna.visualization import plot_rank, plot_parallel_coordinate, plot_param_importances,plot_hypervolume_history,plot_pareto_front, plot_contour, plot_intermediate_values, plot_edf

print(optuna.importance.get_param_importances(study))
params=['sigma','Number of enemies', 'Population size', 'generations']
plot_parallel_coordinate(study, params=params).show()
plot_param_importances(study, params=params).show()
plot_contour(study, params=params).show()
plot_edf(study, params=params).show()
plot_rank(study, params=params).show()
# plot_intermediate_values(study).show()


# plotter.plot_evolution([best_trial.user_attrs['fit_tracker']],show=True)
