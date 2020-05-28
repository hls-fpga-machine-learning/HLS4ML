import numpy as np
import torch
import torch.nn as nn

from model import SegmentClassifier
from estimator import Estimator
from graph import load_graphs, SparseGraph, graph_from_sparse

import os
import sys
filedir = os.path.dirname(os.path.abspath(__file__))
import hls4ml

feature_scale = np.array([1000., np.pi/8, 1000.])
n_features = feature_scale.shape[0]

cuda = False

if cuda:
    np_to_torch = lambda x, volatile=False, dtype=np.float32: (torch.tensor(x.astype(dtype), requires_grad=False).cuda())
else:
    np_to_torch = lambda x, volatile=False, dtype=np.float32: (torch.tensor(x.astype(dtype), requires_grad=False))

if cuda:
    torch_to_np = lambda x: x.cpu().numpy()
else:
    torch_to_np = lambda x: x.detach().numpy()

hidden_dim = 4
n_iters = 1
model = SegmentClassifier(input_dim=n_features, hidden_dim=hidden_dim, n_iters=n_iters)
estim = Estimator(model, loss_func=nn.BCELoss(), cuda=cuda, l1= 0)
#estim.load_checkpoint('model_5_1iteration.pt')
#estim.load_checkpoint('model_4_1iteration.pt')
estim.load_checkpoint('model_3_1iteration.pt')
#estim.load_checkpoint('model_1iteration.pt')

#graph = load_graphs(['graph000001_5.npz'], SparseGraph)
#graph = load_graphs(['graph000001_4.npz'], SparseGraph)
graph = load_graphs(['graph000001_3.npz'], SparseGraph)
#graph = load_graphs(['graph000001.npz'], SparseGraph)

g = graph[0]
print("sparse X",g.X)
print("sparse Ri_rows",len(g.Ri_rows))
print("sparse Ri_cols",len(g.Ri_cols))
print("sparse Ro_rows",len(g.Ro_rows))
print("sparse Ro_cols",len(g.Ro_cols))
print("spase y", g.y)

g = graph_from_sparse(g)

X = np_to_torch(g.X, volatile=False)
Ri = np_to_torch(g.Ri, volatile=False) 
Ro = np_to_torch(g.Ro, volatile=False)

print("X", X)
print("Ri", Ri)
print("Ro", Ro)

var = hls4ml.model.hls_model.WeightVariable('w00', type_name='float', precision='float', data=g.X)
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var, './')
var = hls4ml.model.hls_model.WeightVariable('w01', type_name='int', precision='int', data=g.Ri)
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
var = hls4ml.model.hls_model.WeightVariable('w02', type_name='int', precision='int', data=g.Ro)
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')

os.system('cp firmware/weights/w00.txt tb_data/tb_input_features.dat')
os.system('cp firmware/weights/w01.txt tb_data/tb_adjacency_incoming.dat')
os.system('cp firmware/weights/w02.txt tb_data/tb_adjacency_outgoing.dat')

w1 = model.input_network[0].weight.data.transpose(0,1)
b1 = model.input_network[0].bias.data
print("w1", w1)
print("b1", b1)

var = hls4ml.model.hls_model.WeightVariable('w1', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(w1))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
var = hls4ml.model.hls_model.WeightVariable('b1', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(b1))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')

X = X.view(1, X.shape[0], X.shape[1])
Ri = Ri.view(1, Ri.shape[0], Ri.shape[1])
Ro = Ro.view(1, Ro.shape[0], Ro.shape[1])

H_logits = model.input_network[0](X)
print ("H_logits", H_logits)

H = model.input_network(X)
print ("H", H)

w2 = model.edge_network.network[0].weight.data.transpose(0,1)
b2 = model.edge_network.network[0].bias.data
print("w2", w2)
print("b2", b2)

var = hls4ml.model.hls_model.WeightVariable('w2', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(w2))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
var = hls4ml.model.hls_model.WeightVariable('b2', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(b2))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')

w3 = model.edge_network.network[2].weight.data.transpose(0,1)
b3 = model.edge_network.network[2].bias.data
print("w3", w3)
print("b3", b3)

var = hls4ml.model.hls_model.WeightVariable('w3', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(w3))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
var = hls4ml.model.hls_model.WeightVariable('b3', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(b3))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')

H = torch.cat([H, X], dim=-1)
e_temp = model.edge_network(H, Ri, Ro) 
print("e_temp", e_temp)

w4 = model.node_network.network[0].weight.data.transpose(0,1)
b4 = model.node_network.network[0].bias.data
print("w4", w4)
print("b4", b4)

var = hls4ml.model.hls_model.WeightVariable('w4', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(w4))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
var = hls4ml.model.hls_model.WeightVariable('b4', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(b4))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')

w5 = model.node_network.network[2].weight.data.transpose(0,1)
b5 = model.node_network.network[2].bias.data
print("w5", w5)
print("b5", b5)

var = hls4ml.model.hls_model.WeightVariable('w5', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(w5))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
var = hls4ml.model.hls_model.WeightVariable('b5', type_name='ap_fixed<16,6>', precision='<16,6>', data=torch_to_np(b5))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')

H = model.node_network(H, e_temp, Ri, Ro)
print("H", H)

e = model.forward([X, Ri, Ro]) 
print("e", e)

var = hls4ml.model.hls_model.WeightVariable('w03', type_name='float', precision='float', data=torch_to_np(e))
hls4ml.writer.VivadoWriter.print_array_to_cpp(None,var,'./')
os.system('cp firmware/weights/w03.txt tb_data/tb_output_predictions.dat')
