import json


def load_json(file_path):
    """
    Loads a JSON file into a dictionary.

    :param file_path: Path to the JSON file.
    :return: A dictionary containing the data from the JSON file.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def load_hypertune_params(GA_path='hypertune_params/GA-params.json',
                          CMA_path='hypertune_params/CMA-ES-params.json'):
    GA_params = load_json(GA_path)
    CMA_params = load_json(CMA_path)

    enemy_group_GA = []
    enemy_group_CMA = []

    enemy_groups = {"CMA-ES": enemy_group_CMA, "GA": enemy_group_GA}

    for i in range(1, 9):
        # Remove enemies from params and collect them in a single list
        if GA_params.pop(f"enemy {i}"):
            enemy_group_GA.append(i)
        if CMA_params.pop(f"enemy {i}"):
            enemy_group_CMA.append(i)
    return GA_params, CMA_params, enemy_groups


def evaluate(model, test_env):
    f, p, e, t = test_env.play(pcont=model)
    gain = p - e
    return gain