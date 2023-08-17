import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib import pyplot as plt

class Binary_Graph:
    def __init__(self, num_node):
        self.num_node = num_node
        self.input_name = [f'x_{i}' for i in range(1, num_node)]
        self.nx_seed = 51

        light_red = '#FFD0D0'  # Lighter shade of red
        light_blue = '#D0D0FF'  # Lighter shade of blue

        self.node_colors = [light_red] * len(self.input_name) + [
            light_blue]  # Custom light colors for input nodes and output node
        self.adj = self.generate_adj( num_node)

        self.params = np.random.uniform(low=0.3, high=0.9, size=num_node)
        self.graph = self.generate_graph_from_adjacency(self.adj, self.input_name)

        self.X = np.random.randint(2, size=num_node)


        self.max_possible_X = np.sum(self.adj, axis=0)
        self.max_possible_X[self.max_possible_X ==0] = 1


        self.prob = None
        self.belief_x = np.zeros(num_node)

        self.belief_list = None

        self.all_descendants = self.__get_all_descendants()


    def generate_adj(self, n):
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = np.random.randint(2)

        for i in range(n):
            if np.sum(matrix[i,:]) == 0 and np.sum(matrix[:, i]) == 0 :
                return  self.generate_adj(n)
            else:
                continue
        return matrix

    def reset(self):

        X = np.random.binomial(1, p = self.params)


        #empty intervention to sample a context
        self.context, _ = self.intervention(X, None, None)
        #print("reset context : ", self.context)



        return self.context

    def load(self, dic_path):
        with open(dic_path, 'r') as file:
            dic = json.load(file)
        self.params = np.array(dic['params'])
        self.adj = np.array(dic['adj'])
        self.all_descendants = dic['all_descendants']
        self.graph = self.generate_graph_from_adjacency(self.adj, self.input_name)

        self.X = np.random.randint(2, size=self.num_node)

        self.max_possible_X = np.sum(self.adj, axis=0)
        self.max_possible_X[self.max_possible_X == 0] = 1

        self.prob = None
        self.belief_x = np.zeros(self.num_node)

        self.belief_list = None

        self.all_descendants = self.__get_all_descendants()



    def generate_graph_from_adjacency(self, adjacency_matrix, input_name, output_name="Y"):
        num_nodes = self.num_node
        graph = nx.DiGraph()

        # Add input nodes
        for node_name in input_name:
            graph.add_node(node_name)

        # Add output node
        graph.add_node(output_name)

        #
        # Add edges
        for i in range(num_nodes):
            for j in range(num_nodes-1):
                if adjacency_matrix[i, j] != 0:
                    graph.add_edge(input_name[i], input_name[j])

            #add edge to target
            if adjacency_matrix[i, num_nodes-1] != 0:
                graph.add_edge(input_name[i], output_name)


        return graph

    def __get_descendants(self, adj_matrix, node_index):
        descendants = []
        children = np.nonzero(adj_matrix[node_index])[0]

        for child in children:

            descendants.append(child)
            descendants.extend(self.__get_descendants(adj_matrix, child))

        return descendants

    def __get_all_descendants(self):
        all_descendants = []
        for i in range(self.num_node):
            all_descendants.append(list(set(self.__get_descendants(self.adj, i))))

        return all_descendants


    def show(self):
        nx.draw(self.graph, with_labels=True, node_color=self.node_colors)
        plt.show()

    # def step(self, do_idx = None):
    #     # probability of P(instance_X_i  = 1 | Pa(X_i) ) is calculate by
    #     # value of X_i / max possible value of x_i
    #     if do_idx is None:
    #         self.prob = self.X / self.max_possible_X * self.params
    #         self.instance_x = np.random.binomial(1, p = self.prob)
    #
    #     else:
    #
    #         self.prob = self.X / self.max_possible_X * self.params
    #         print( "do idx : ", do_idx, " " ,self.prob)
    #         new_instance = np.random.binomial(1, p=self.prob)
    #         # only change the distribution which related to
    #         self.instance_x = self.context.copy()
    #         self.instance_x[self.all_descendants[do_idx]] = new_instance[do_idx]
    #         # modify nodo of intervned
    #         self.instance_x[do_idx] = 1
    #
    #     return self.instance_x




    def intervention(self, node_value,do_idx, do_value):
        """
        iterate over topological sort,
        for example the parents of x_3 cannot be x_6.
        So the complexity in O(n^2)

        First get the value of each node, then sample from the distribution
        get binary value

        :param do_idx: index of node to DO
        :param do_value:
        :return: new observation after intervention
        """
        X = node_value.copy()

        belief_list = []
        for i in range(self.num_node):
            # pass the node has no parents


            # set value of intervention node
            if i == do_idx:

                X[i] = do_value

                belief_list.append(do_value)

            # no incoming edge to node i
            elif np.sum(self.adj[:, i]) == 0 :
                belief_list.append(X[i])


            # leave the subgraph that not changed by do_x_i
            elif  do_idx is not None and i not in self.all_descendants[do_idx]:
                belief_list.append(X[i])
            else:
                # else set the node to be 0, and update the values given by its parents

                belief_xi = np.matmul(belief_list[:i], self.adj[:i,i]) / self.max_possible_X[i] * self.params[i]


                prob_xi = np.matmul(X[:i], self.adj[:i,i]) / self.max_possible_X[i] * self.params[i]

                X[i] = np.random.binomial(1, p=prob_xi)
                belief_list.append(belief_xi)


        # last belief is expected reward
        # instance_x = np.random.binomial(1, p=belief_list)
        expected_reward = belief_list[-1]
        self.X = X

        self.belief_list = np.round(belief_list, 2)
        return X, expected_reward

    def show_value(self, value = None):
        pos = nx.spring_layout(self.graph, seed = self.nx_seed)  # Position nodes using a spring layout

        # Draw nodes with labels and values
        if value is not None:

            labels = {node: f"{node}\n{value[i]}" for i, node in enumerate(self.graph.nodes())}
        else:
            labels = {node: f"{node}\n{self.X[i]}" for i, node in enumerate(self.graph.nodes())}
        nx.draw_networkx_nodes(self.graph, pos, node_color=self.node_colors)
        nx.draw_networkx_labels(self.graph, pos, labels=labels)

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos)

        plt.axis('off')
        plt.show()

    def search_optimal(self, context):

        expected_reward_list = []

        self.dist_Y = []
        #self.intervened_dist_x = []
        context_copy = context.copy()

        for i in range(self.num_node - 1):
            context_copy = context.copy()



            #self.show_value()
            # store node value of Y to calculate distribution of reward

            intervened_X, Y_hat = self.intervention(context_copy, i, 1)

            #self.intervened_dist_x.append(intervened_X.copy())

            self.dist_Y.append(Y_hat.copy())




        self.regret_list = np.max(self.dist_Y) - np.array(self.dist_Y)
        # print("regret : ",self.regret_list)
        # print(self.intervened_dist_x)
        return self.regret_list

    def calculate_permutation_probability(self, permutation):
        """
        Calculate the probability of a specific permutation given the causal structure and parameters.

        :param permutation: A list representing a specific permutation of node values.
        :return: Probability of the permutation.
        """
        prob = 1.0  # Starting value for probability.

        # Ensure that the permutation length matches the number of nodes.
        if len(permutation) != self.num_node:
            raise ValueError("The length of the permutation does not match the number of nodes.")

        # Traverse the graph in a sequence (assuming topological order based on node indices).
        for i in range(self.num_node):

            if np.sum(self.adj[:, i]) == 0 :
                prob_xi = self.params[i]
            else:
                prob_xi = np.matmul(permutation[:i], self.adj[:i, i]) / self.max_possible_X[i] * self.params[i]

            if permutation[i] == 0:
                prob *= (1 - prob_xi)
            else:
                prob *= prob_xi

        return prob

# bg = Binary_Graph(6)
#
# cxt = bg.reset()
# x, _ = bg.intervention(cxt, 1,1)
# print(bg.belief_list)
# bg.show_value(bg.belief_list)
# bg.show_value(x)
# print(bg.params)
# print(bg.max_possible_X)
