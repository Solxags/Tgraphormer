import numpy as np
import os
import torch
from tqdm import tqdm

from model import make_model
from utils import gen_attention_bias,gen_node_bias,load_pickle
import utils

model=make_model(2,40,12,2,0.1,80,2,5,207,80,1)

# data = np.load(os.path.join('data', 'METR-LA', 'train.npz'))
# inputs= data['x']
# targets= data['y']
# input=inputs[0:2][..., :2]
# target=targets[0:2][..., :1]
# print(input.shape,target.shape) 
# input=torch.tensor(input, dtype=torch.float32)
# target=torch.tensor(target, dtype=torch.float32)


datasets = utils.get_datasets('METR-LA', 2, 1)
data_loaders = utils.get_dataloaders(datasets, 64)


scaler = utils.ZScoreScaler(datasets['train'].mean , datasets['train'].std )

_, _, adj_mx = load_pickle(os.path.join('data', 'sensor_graph', 'adj_mat.pkl'))
in_degree,out_degree=gen_node_bias(adj_mx)
shortest_path_result,shortest_hop_result=gen_attention_bias(adj_mx)
in_degree=torch.tensor(in_degree, dtype=torch.long)
out_degree=torch.tensor(out_degree, dtype=torch.long)
shortest_hop_result=torch.tensor(shortest_hop_result, dtype=torch.long)
shortest_path_result=torch.tensor(shortest_path_result, dtype=torch.float32)
print("run")
opt=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=True)
for x, y in tqdm(data_loaders['train'], f'{"train".capitalize():5} {0}'):
    input = scaler.transform(x, 0.0)
    result=model(input,in_degree,out_degree,shortest_path_result,shortest_hop_result)
    print(utils.utils.get_number_of_parameters(model))
    result=scaler.inverse_transform(result, 0.0)

    print(result.shape)

    loss=utils.get_loss('MaskedMAELoss')
    loss_ = loss(result, y)

    print(loss_)

    loss_.backward()

    opt.step()