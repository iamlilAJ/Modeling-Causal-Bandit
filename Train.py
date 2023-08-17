import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import CausalInference
from utils import adj_matrix_to_edges, remove_self_child, get_descendants, rerank_and_transform, add_interventional_data, add_observational_data
from learn_bayesian_network import Bayesian_Model_Bandit
from LinUCB_Algo import LinUCB
from binary_value_graph import Binary_Graph
from matplotlib import pyplot as plt
from tqdm import  tqdm
import copy

class Trainer():
    def __init__(self, model : Bayesian_Model_Bandit, algo : LinUCB):
        self.model = model
        self.algo = algo
        self.log = None

    def train(self, n_rounds = 500, show_progress = False):
        iterable = range(n_rounds)
        if show_progress:
            iterable = tqdm(iterable)
        for _ in iterable:
            context = self.model.generate_context()
            #transfrom dic to list
            context_list = rerank_and_transform(context)

            selected_arm = self.algo.select_arm(context_list)
            intervened_dist, reward = self.model.intervention(selected_arm, context)

            self.algo.update(selected_arm, context_list, reward)

class Tester():
    def __init__(self, env : Binary_Graph, algo : LinUCB, model : Bayesian_Model_Bandit ,t, train_rounds):
        self.env = env
        self.model = model
        self.algo = copy.copy(algo)
        self.online_algo = copy.copy(algo)
        self.log = None
        self.reward_list = None
        self.regret_list = None
        self.online_reward_list = None
        self.online_regret_list = None
        self.num_node = env.num_node
        self.T = t
        self.train_rounds = train_rounds

    def load_dic(self, dic_path):
        self.env.load(dic_path)

    def test(self, n_rounds = 500, compare = False ,online_update = False,  given_data = True, show_progress = True):
        ucb_reward_list = []
        ucb_regret_list = []
        sum_reward = 0
        sum_regret = 0

        online_ucb_reward_list = []
        online_ucb_regret_list = []
        online_sum_reward = 0
        online_sum_regret = 0

        online_context_list = []
        online_intervened_list = []
        online_action_list = []

        if compare:
            self.baseline_regret_list = []
            self.baseline_reward_list = []
            sum_reward_b = 0
            sum_regret_b = 0
            self.baseline = LinUCB(self.algo.num_arms,
                                         self.algo.num_features, self.algo.alpha)

        iterable = range(n_rounds)
        if show_progress :
            iterable = tqdm(iterable)

        for t in iterable:
            context = self.env.reset().copy()
            regret_list = self.env.search_optimal(context.copy())

            if given_data:
                ucb_arm = self.algo.select_arm(context)
                intervened_context,_ = self.env.intervention(  context.copy(), ucb_arm,1)
                reward = intervened_context[-1]
                self.algo.update(ucb_arm, context, reward)


                regret = regret_list[ucb_arm]

                sum_regret += regret
                sum_reward += reward
                ucb_reward_list.append(sum_reward)
                ucb_regret_list.append(sum_regret)
            #else not data given, we will not update this algorithm and using online update instead



            if compare :

                baseline_arm = self.baseline.select_arm(context)
                intervened_context, _ = self.env.intervention( context.copy(), baseline_arm, 1)
                reward_b = intervened_context[-1]
                self.baseline.update(baseline_arm, context, reward_b)

                regret_b = regret_list[baseline_arm]

                sum_regret_b += regret_b
                sum_reward_b += reward_b
                self.baseline_reward_list.append(sum_reward_b)
                self.baseline_regret_list.append(sum_regret_b)

            if online_update:
                online_ucb_arm = self.online_algo.select_arm(context)
                intervened_context, _ = self.env.intervention( context.copy(), online_ucb_arm, 1)
                reward_online = intervened_context[-1]
                self.online_algo.update(online_ucb_arm, context.copy(), reward_online)

                regret_online = regret_list[online_ucb_arm]

                online_sum_regret += regret_online
                online_sum_reward += reward_online
                online_ucb_reward_list.append(online_sum_reward)
                online_ucb_regret_list.append(online_sum_regret)

                online_context_list.append(context)
                online_intervened_list.append(intervened_context)
                online_action_list.append(online_ucb_arm)

            # every t round fit the bayesian model and retrain algorithm
            if t % self.T == 0 and t != 0 and online_update:
                data = add_observational_data(self.model.data, online_context_list)
                data2 = add_interventional_data(data, online_intervened_list, online_action_list, self.env.all_descendants)
                self.model.data = data2
                self.model.learn()

                trainer = Trainer(self.model, self.online_algo)
                trainer.train(self.train_rounds)

                online_context_list = []
                online_intervened_list = []
                online_action_list = []
                self.online_algo = copy.copy(trainer.algo)

        self.reward_list = ucb_reward_list
        self.regret_list = ucb_regret_list


        self.online_reward_list = online_ucb_reward_list
        self.online_regret_list = online_ucb_regret_list

    def plot_reward(self):
        plt.plot(self.reward_list, color="green")

    def plot_regret(self):
        plt.plot(self.regret_list, color = 'yellow')

    def plot_compare(self):
        # Create a 1x2 grid for subplots
        plt.subplot(1, 2, 1)

        # Plot the first subplot (reward comparison)
        plt.plot(self.reward_list, color="green", label="Model-Based Reward")
        plt.plot(self.baseline_reward_list, color="blue", label="Baseline Reward")
        plt.plot(self.online_reward_list, color = 'yellow', label = 'online update reward')
        plt.xlabel("Round")
        plt.ylabel("Reward")
        plt.legend()

        # Create the second subplot
        plt.subplot(1, 2, 2)

        # Plot the second subplot (regret comparison)
        plt.plot(self.regret_list, color="green", label="Model-Based Regret")
        plt.plot(self.baseline_regret_list, color="blue", label="Baseline Regret")
        plt.plot(self.online_regret_list, color='yellow', label='online update Regret')
        plt.xlabel("Round")
        plt.ylabel("Regret")
        plt.legend()

        # Show the plots
        plt.show()


# class Tester():
#     def __init__(self, env : Binary_Graph, algo : LinUCB):
#         self.env = env
#         self.algo = algo
#         self.log = None
#         self.reward_list = None
#         self.regret_list = None
#         self.num_node = env.num_node
#
#     def load_dic(self, dic_path):
#         self.env.load(dic_path)
#
#     def test(self, n_rounds = 500, compare = False ):
#         ucb_reward_list = []
#         ucb_regret_list = []
#         sum_reward = 0
#         sum_regret = 0
#         if compare:
#             self.baseline_regret_list = []
#             self.baseline_reward_list = []
#             sum_reward_b = 0
#             sum_regret_b = 0
#             self.baseline = LinUCB(self.algo.num_arms,
#                                          self.algo.num_features, self.algo.alpha)
#
#         for t in tqdm(range(n_rounds)):
#             context = self.env.reset().copy()
#             ucb_arm = self.algo.select_arm(context)
#             intervened_context,_ = self.env.intervention(  context.copy(), ucb_arm,1)
#             reward = intervened_context[-1]
#             self.algo.update(ucb_arm, context, reward)
#
#             regret_list = self.env.search_optimal(context)
#             regret = regret_list[ucb_arm]
#
#             sum_regret += regret
#             sum_reward += reward
#             ucb_reward_list.append(sum_reward)
#             ucb_regret_list.append(sum_regret)
#
#             if compare :
#
#                 baseline_arm = self.baseline.select_arm(context)
#                 intervened_context, _ = self.env.intervention( context.copy(), baseline_arm, 1)
#                 reward_b = intervened_context[-1]
#                 self.baseline.update(baseline_arm, context, reward_b)
#
#                 regret_b = regret_list[baseline_arm]
#
#                 sum_regret_b += regret_b
#                 sum_reward_b += reward_b
#                 self.baseline_reward_list.append(sum_reward_b)
#                 self.baseline_regret_list.append(sum_regret_b)
#
#
#         self.reward_list = ucb_reward_list
#         self.regret_list = ucb_regret_list
#
#     def plot_reward(self):
#         plt.plot(self.reward_list, color="green")
#
#     def plot_regret(self):
#         plt.plot(self.regret_list, color = 'yellow')
#
#     def plot_compare(self):
#         # Create a 1x2 grid for subplots
#         plt.subplot(1, 2, 1)
#
#         # Plot the first subplot (reward comparison)
#         plt.plot(self.reward_list, color="green", label="Model-Based Reward")
#         plt.plot(self.baseline_reward_list, color="blue", label="Baseline Reward")
#         plt.xlabel("Round")
#         plt.ylabel("Reward")
#         plt.legend()
#
#         # Create the second subplot
#         plt.subplot(1, 2, 2)
#
#         # Plot the second subplot (regret comparison)
#         plt.plot(self.regret_list, color="red", label="Model-Based Regret")
#         plt.plot(self.baseline_regret_list, color="purple", label="Baseline Regret")
#         plt.xlabel("Round")
#         plt.ylabel("Regret")
#         plt.legend()
#
#         # Show the plots
#         plt.show()

















