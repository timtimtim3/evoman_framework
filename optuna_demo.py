import optuna
import pymysql
pymysql.install_as_MySQLdb()


study = optuna.create_study(study_name='test', direction="maximize", storage='mysql+pymysql://root:UbuntuMau@localhost/example', load_if_exists=True)

best_trial = study.best_trial
print(best_trial.user_attrs["enemies"])

print("  Value: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))