import numpy as np
import pandas as pd
from CAUSAL.causal_bandit.model.binary_value_graph import  Binary_Graph
import os
import json
np.random.seed(51)
config = {
    "node_number" : [5, 8, 10],
    "n_sample" : [5, 8, 10] ,
    "file_dir" : "/Users/liuanjie/PycharmProjects/pythonProject19/CAUSAL/causal_bandit/model/data"
}
def save_matrix_to_txt(matrix, filename):
    with open(filename, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
def save_env(env, file_name):


    dict = {

        'adj' : env.adj.tolist(),
        'params' :  [float(param) for param in env.params.tolist()],
        'all_descendants' : env.all_descendants
    }
    print(dict)


    with open(file_name, 'w') as file:
        json.dump(dict, file, default=np_encoder)

def generator():

    file_dir = config['file_dir']
    file_format = 'csv'
    for i,node_number in enumerate(config['node_number']):
        env = Binary_Graph(node_number)
        name_node = env.input_name
        name_node.append('Y')
        env.reset()

        causal_structure = env.adj

        file_name =os.path.join(file_dir, f"{env.num_node}_node_env.json")
        save_env(env,file_name)
        sample = []
        for _ in range( 200 * config['n_sample'][i]):
            cxt = env.reset()
            sample.append(cxt)

        array_sample = np.array(sample)
        df = pd.DataFrame(array_sample, columns=name_node)

        file_name = os.path.join(file_dir, f'{node_number}_nodes.{file_format}')
        df.to_csv(file_name, index=False)

        file_name = os.path.join(file_dir, f"{node_number}_node_matrix.txt")
        save_matrix_to_txt(causal_structure, file_name)





generator()




