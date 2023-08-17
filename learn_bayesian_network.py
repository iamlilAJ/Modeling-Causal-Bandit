import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.inference import CausalInference
from utils import adj_matrix_to_edges, remove_self_child
import os


data = pd.read_csv('data/5_nodes.csv')
causal_structure =np.loadtxt('data/5_node_matrix.txt')



class Bayesian_Model_Bandit(BayesianNetwork):
    def __init__(self, node_number):
        # Call the __init__ method of the parent class
        super().__init__()

        self.context = None
        self.node_number = node_number
        self.node_name = self.input_name = [f'x_{i}' for i in range(1, node_number)]
        self.node_name.append("Y")

    def collect_data(self, data_path='data/'):
        path = os.path.join(data_path, f'{self.node_number}_nodes.csv')
        self.data = pd.read_csv(path)

        path = os.path.join(data_path, f'{self.node_number}_node_matrix.txt')
        self.causal_structure = np.loadtxt(path)

    def learn(self, if_bayesian_estimator = False):
        edges = adj_matrix_to_edges(self.causal_structure)
        self.add_edges_from(edges)
        if if_bayesian_estimator:
            self.fit(self.data, estimator=BayesianEstimator, prior_type="BDeu", complete_samples_only=False)
        else:
            self.fit(self.data, estimator=MaximumLikelihoodEstimator, complete_samples_only= False )

    def generate_context(self):
        self.context = self.simulate(n_samples=1, show_progress=False)

        return  self.context.iloc[0].to_dict()

    def intervention(self, do_idx, context, n_samples = 1):
        node = self.node_name[do_idx]

        evidence = remove_self_child(self, node, context)


        if n_samples > 1 :
            intervened_dist = self.simulate(n_samples=n_samples, do={node: 1}, evidence=evidence, show_progress=False)
            return intervened_dist
        else:

            intervened_dist = self.simulate( n_samples=1, do= {node : 1},evidence= evidence, show_progress=False)
            intervened_dist = intervened_dist.iloc[0].to_dict()
            reward = intervened_dist['Y']
            return intervened_dist, reward















