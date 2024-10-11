import optuna
import pymysql
pymysql.install_as_MySQLdb()
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import plotter



study = optuna.create_study(study_name='run1overnight', direction="maximize", storage='mysql+pymysql://root:UbuntuMau@localhost/CMAES', load_if_exists=True)

best_trial = study.best_trial
print(best_trial.user_attrs["enemies"])

print("  Value: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))


controller = np.array(best_trial.user_attrs['best_model_solution'])


# env2 = Environment(experiment_name='test',
#                       enemies=[1,2,3,4,5,6,7,8],
#                       playermode="ai",
#                       multiplemode="yes",
#                       player_controller=player_controller(10),
#                       enemymode="static",
#                       level=2,
#                       speed="normal",
#                       visuals=True)

# env = Environment(experiment_name='test',
#                       enemies=[1],
#                       playermode="ai",
#                     #   multiplemode="yes",
#                       player_controller=player_controller(10),
#                       enemymode="static",
#                       level=2,
#                       speed="fastest",
#                       visuals=True)
# enemies = [1,2,3,4,5,6,7,8]
# for enemy in enemies:
#     env.update_parameter('enemies',[enemy])
#     print(env.play(pcont=controller))

print(optuna.importance.get_param_importances(study))

plotter.plot_evolution([best_trial.user_attrs['fit_tracker']],show=True)
