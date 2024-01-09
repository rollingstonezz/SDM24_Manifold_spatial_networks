import os
import numpy as np
from tqdm import tqdm
import time

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import contains_isolated_nodes

import pyvista as pv
from pyvista import examples

import pickle
import numpy.linalg as LA
from torch import Tensor

from model import *

def S_eccentricity(dis_matrix):
    eccentricity = dis_matrix.max(axis=1)
    return eccentricity
def S_diameter(dis_matrix):
    eccentricity = S_eccentricity(dis_matrix)
    return np.max(eccentricity)
def S_radius(dis_matrix):
    eccentricity = S_eccentricity(dis_matrix)
    return np.min(eccentricity)


def get_angle(v1: Tensor, v2: Tensor) -> Tensor:
    return torch.atan2(
        torch.cross(v1, v2, dim=1).norm(p=2, dim=1), (v1 * v2).sum(dim=1))

def calculate_geometric_features(item):
    # traj dist
    dist_to_A = Tensor(LA.norm(item-item[0], axis=-1))
    dist_to_B = Tensor(LA.norm(item-item[-1], axis=-1))

    l_iiplusone = Tensor(item[:-1] - item[1:])
    dist = Tensor(LA.norm(l_iiplusone, axis=-1))

    # traj angles theta
    AB_vector = Tensor(item[-1] - item[0]).view(1,-1)
    theta = get_angle(l_iiplusone, AB_vector.repeat(l_iiplusone.shape[0],1))

    # traj angles phi
    l_iB = Tensor(item[:-1] - item[-1])
    l_iplusoneA = Tensor(item[1:] - item[0])
    n_iiplusoneB = torch.cross(l_iiplusone, l_iB) 
    n_iiplusoneA = torch.cross(l_iiplusone, l_iplusoneA) 
    phi_B = get_angle(n_iiplusoneB[:-1], n_iiplusoneB[1:])
    phi_iiplusoneB = torch.cat(
            [
                torch.cat([Tensor([0]), phi_B]).view(-1,1),
                torch.cat([phi_B, Tensor([0])]).view(-1,1)
            ]
            ,dim=1
        )
    phi_A = get_angle(n_iiplusoneA[:-1], n_iiplusoneA[1:])
    phi_iiplusoneA = torch.cat(
            [
                torch.cat([Tensor([0]), phi_A]).view(-1,1),
                torch.cat([phi_A, Tensor([0])]).view(-1,1)
            ]
            ,dim=1
        )
    
    # concat together
    geometric_features = torch.cat([dist.view(-1,1), theta.view(-1,1), phi_iiplusoneB, phi_iiplusoneA],dim=-1)
    
    return geometric_features


    
if __name__ == '__main__':
    graphs, paths = [], []
    for surface in ['sin','cos','poly','sphere','exp']:
        with open('../data/synthetic/graphs_'+surface+'.pkl', 'rb') as file:
            graphs += pickle.load(file) 
        with open('../data/synthetic/paths_'+surface+'.pkl', 'rb') as file:
            paths += pickle.load(file)

    dataset = {'train':[], 'test':[]}
    traj_features = {'train':[], 'test':[]}
    num_graphs = len(graphs)
    for i in tqdm(range(num_graphs)):

        if i < int(0.8*num_graphs):
            traintest = 'train'
        else:
            traintest = 'test'

        seq_lengths = []
        for t, edge_index in enumerate(graphs[i].edge_index.numpy().T):
            item = paths[i][t]
            seq_lengths.append(len(item))

        padded_traj = []   
        for t, edge_index in enumerate(graphs[i].edge_index.numpy().T):
            item = paths[i][t]
            # compute traj geometric features
            if len(item) - 1 > 1:
                geometric_features = calculate_geometric_features(item)
            else:
                geometric_features = torch.zeros((1, 6)) # extreme situations        
            temp = torch.cat([geometric_features, torch.zeros((np.max(seq_lengths) - len(geometric_features), 6))], dim=0)
            padded_traj.append(temp)

        padded_traj = torch.cat(padded_traj).view(-1, np.max(seq_lengths), 6)
        traj_tensor = torch.tensor(padded_traj, dtype=torch.float)    

        lstm_traj = torch.nn.utils.rnn.pack_padded_sequence(
            traj_tensor, 
            seq_lengths, 
            batch_first=True,
            enforce_sorted=False
        )

        traj_features[traintest].append(lstm_traj)
        dataset[traintest].append(graphs[i])
        
    # model
    model = GIN(3, 64, 1)

    # optimizer setting
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # send to gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    lstm_traj = lstm_traj.to(device)

    def train(loader, traj_features, y_index):
        model.train()
        total_loss = []
        for i, data in enumerate(loader):
            optimizer.zero_grad()  # Clear gradients.
            x, edge_index, y = data.x, data.edge_index, data.y[y_index:y_index+1]
            x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
            lstm_traj = traj_features[i]
            lstm_traj = lstm_traj.to(device)
            out = model(x, edge_index, lstm_traj)  # Perform a single forward pass.
            loss = criterion(out,y)  # Compute the loss solely based on the training nodes.
            total_loss.append(float(loss)) 
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
        return np.mean(total_loss)

    def test(loader, traj_features, y_index):
        model.eval()
        pred_list = []
        y_list = []
        for i, data in enumerate(loader):
            x, edge_index, y = data.x, data.edge_index, data.y[y_index:y_index+1]
            x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
            lstm_traj = traj_features[i]
            lstm_traj = lstm_traj.to(device)
            out = model(x, edge_index, lstm_traj)  # Perform a single forward pass.
            pred = out
            pred_list.append(pred.detach().cpu())
            y_list.append(y.detach().cpu())
        y_true = torch.cat(y_list)
        y_pred = torch.cat(pred_list)
        mse = criterion(y_true, y_pred)
        return mse, y_pred, y_true

    print('Epoch, Loss, Train Accuracy,  Test Accuracy')
    num_epochs = 100
    for y_index in [0,1,2]:
        print('y_index: ', y_index)
        start_time = time.time()
        for epoch in (range(1, num_epochs+1)):                
            # train model
            train_mse = train(dataset['train'], traj_features['train'], y_index)
            # test model
            #train_mse, train_y_pred, train_y_pred = test(dataset['train'], traj_features['train'])
            test_mse, test_y_pred, test_y_true = test(dataset['test'], traj_features['test'], y_index)

            print(f'{epoch:03d}, {np.sqrt(train_mse):.4f},  {np.sqrt(test_mse):.4f},',\
                  "--- %s seconds ---" % (time.time() - start_time))
