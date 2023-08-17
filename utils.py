import pandas as pd
import numpy as np
def adj_matrix_to_edges(adj_matrix):
    num_nodes = adj_matrix.shape[0]
    nodes = [f'x_{i}' for i in range(1, num_nodes)]
    nodes.append('Y')

    edges = []
    for i in range(num_nodes - 1):
        for j in range(num_nodes):
            if adj_matrix[i, j] != 0:
                edges.append((nodes[i], nodes[j]))

    return edges

def get_descendants(model, node):
    descendants = []

    def dfs(node):
        children = model.get_children(node)
        descendants.extend(children)
        for child in children:
            dfs(child)

    dfs(node)
    return list(set(descendants))


def remove_self_child(model, node, context):

    descendants = get_descendants(model, node)
    descendants.append(node)

    for node in descendants:
        context.pop(node)

    return  context

def rerank_and_transform(input_dict):
    # Re-rank the dictionary keys
    reranked_keys = sorted(input_dict.keys(), key=lambda x: int(x[2:]) if x != 'Y' else float('inf'))
    # The lambda function sorts keys 'x1', 'x2', ..., 'xN' based on their numeric suffix, while placing 'Y' at the end

    # Transform the re-ranked dictionary to a list
    reranked_list = [input_dict[key] for key in reranked_keys]

    return reranked_list



def add_observational_data(prev_data, new_data):
    num_node = len(new_data[0])
    columns = [f"x_{i + 1}" for i in range(num_node - 1)] + ["Y"]

    df = pd.DataFrame(new_data, columns=columns)
    complete_data = pd.concat([prev_data, df], ignore_index=True)
    return complete_data



def add_interventional_data(prev_data, new_data, history_action, all_descendants):
    num_nodes = len(new_data[0])
    num_rows = len(history_action)

    result = np.full((num_rows, num_nodes), np.nan)

    for row_idx, action in enumerate(history_action):
        intervened_nodes = all_descendants[action]
        for idx, node in enumerate(intervened_nodes):
            result[row_idx, node] = new_data[row_idx][node]

    columns = [f"x_{i + 1}" for i in range(num_nodes - 1)] + ["Y"]
    df = pd.DataFrame(result, columns=columns)
    complete_data = pd.concat([prev_data, df], ignore_index=True)
    return complete_data









# Example usage:
# Assuming `model` is your Bayesian network model and `node` is the node from which you want to remove children
# and `context` is the dictionary containing all nodes.
# For example, if you want to remove children of 'node1' except for 'node2':
# context = {'node1': [child1, child2, child3, ...], 'node2': [child4, child5, ...], ...}
# context = remove_child(model, 'node1', context)


