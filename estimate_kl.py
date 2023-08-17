from learn_bayesian_network import Bayesian_Model_Bandit
from binary_value_graph import Binary_Graph
import numpy as np
import os
from matplotlib import pyplot as plt

def kl(p, q):
    return p * np.log(p / q)

def generate_binary_combinations(n):
    return [[int(bit) for bit in format(i, '0' + str(n) + 'b')] for i in range(2 ** n)]

def generate_dict(keys, values):
    return {k: v for k, v in zip(keys, values)}

def get_prob(num_node, n_data):
    dic_path = os.path.join('/Users/liuanjie/PycharmProjects/pythonProject19/CAUSAL/causal_bandit/model/data/',
                            f'{num_node}_node_env.json')
    env = Binary_Graph(num_node)
    env.load(dic_path)

    bmb = Bayesian_Model_Bandit(num_node)
    bmb.collect_data()
    bmb.data = bmb.data.iloc[:n_data]
    bmb.learn()

    permunation = generate_binary_combinations(num_node)
    name_node = env.input_name
    name_node.append('Y')

    res = 0
    eps = 1e-8
    for node_value in permunation:
        p = env.calculate_permutation_probability(node_value)
        dic = generate_dict(name_node, node_value)
        q = bmb.get_state_probability(dic)

        res += kl(p + eps, q + eps)

    return res


def plot_kl_vs_data_separate():
    num_node = [5, 8, 10]
    interval = 100


    for node in num_node:
        plt.figure(figsize=(10, 6))

        file_path = os.path.join('/Users/liuanjie/PycharmProjects/pythonProject19/CAUSAL/causal_bandit/model/data/',
                                 f'{node}_nodes.csv')
        max_data = sum(1 for _ in open(file_path)) - 1  # Subtracting 1 to account for the header

        data_sizes = list(range(interval, max_data + 1, interval))
        kl_values = [get_prob(node, n) for n in data_sizes]

        plt.plot(data_sizes, kl_values)

        plt.xlabel("Number of Data")
        plt.ylabel("KL Divergence")
        plt.title(f"KL Divergence vs Number of Data for {node} Nodes")
        plt.grid(True)
        plt.show()


#plot_kl_vs_data_separate()


def plot_kl_vs_data():
    num_node = [5, 8, 10]
    interval = 100
    max_data = 1000  # Assume a max data size for visualization

    plt.figure(figsize=(10, 6))

    for node in num_node:
        data_sizes = list(range(interval, max_data + 1, interval))
        kl_values = [get_prob(node, n) for n in data_sizes]

        plt.plot(data_sizes, kl_values, label=f'{node} nodes')

    plt.xlabel("Number of Data")
    plt.ylabel("KL Divergence")
    plt.legend()
    plt.title("KL Divergence vs Number of Data for Different Node Sizes")
    plt.grid(True)
    plt.show()

# Run the plotting function
plot_kl_vs_data()



