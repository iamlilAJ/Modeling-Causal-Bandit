from Train import Trainer, Tester
from LinUCB_Algo import  LinUCB
from learn_bayesian_network import Bayesian_Model_Bandit
from binary_value_graph import  Binary_Graph
import os
import yaml
import copy
import numpy as np
from tqdm import tqdm
np.random.seed(42)
with open('config.yaml', 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

def run_exp(config):
    print(config)
    baseline_regret_list = []
    mb_regret_list = []
    online_mb_regret_list = []

    num_node = config['num_node']
    train_rounds = config['train_rounds']
    test_rounds = config['test_rounds']
    n_rounds = config['n_rounds']
    T = config['T']
    online_train_rounds = config['online_train_rounds']
    num_data = config['num_data']
    alpha = config['alpha']

    bmb = Bayesian_Model_Bandit(num_node)
    bmb.collect_data()
    #bmb.data = bmb.data.sample(n = num_data)
    bmb.data = bmb.data.iloc[:num_data]
    print(bmb.data)
    ucb = LinUCB(num_arms=bmb.node_number - 1, num_features=bmb.node_number - 1, alpha=alpha)
    if num_data == 0 :
        given_data = False

    else:
        bmb.learn()
        print(bmb.data)
        given_data = True

    for _ in tqdm(range(n_rounds)) :
        if given_data:
            trainer = Trainer(bmb, ucb)
            trainer.train(n_rounds=train_rounds)
            algo = copy.copy(trainer.algo)
        else:
            algo = LinUCB(num_arms=bmb.node_number - 1, num_features=bmb.node_number - 1, alpha=alpha)

        #print(algo.theta)

        env = Binary_Graph(num_node=num_node)

        dic_path = os.path.join('/Users/liuanjie/PycharmProjects/pythonProject19/CAUSAL/causal_bandit/model/data/',
                                f'{num_node}_node_env.json')
        test = Tester(env, algo, bmb, T, online_train_rounds)

        test.load_dic(dic_path=dic_path)
        test.test(n_rounds=test_rounds, compare=True, online_update= True, given_data= given_data,show_progress=False)
        baseline_regret_list.append(test.baseline_regret_list)
        mb_regret_list.append(test.regret_list)
        online_mb_regret_list.append(test.online_regret_list)

    return baseline_regret_list, mb_regret_list, online_mb_regret_list








