import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
from torch.nn import Embedding, Sequential, Linear, ModuleList, ReLU
from torch_geometric.nn.inits import reset
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GINConv
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from typing import Callable, Optional, Union

from torch_scatter import scatter

class GINEConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, nn2: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.nn2 = nn2
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        if edge_dim is not None:
            if isinstance(self.nn, torch.nn.Sequential):
                nn = self.nn[0]
            if hasattr(nn, 'in_features'):
                in_channels = nn.in_features
            elif hasattr(nn, 'in_channels'):
                in_channels = nn.in_channels
            else:
                raise ValueError("Could not infer input channels from `nn`.")
            self.lin = Linear(edge_dim, in_channels)

        else:
            self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        return self.nn(out)


    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        temp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.nn2(temp)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'
    
class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=2, random_seed=1234567):
        super(MLP, self).__init__()
        torch.manual_seed(random_seed)
        self.lin_list = ModuleList()
        if num_layers == 1:
            self.lin_list.append(Linear(input_channels, output_channels, bias=False))
        else:
            self.lin_list.append(Linear(input_channels, hidden_channels, bias=False))
            for _ in range(num_layers-2):
                self.lin_list.append(Linear(hidden_channels, hidden_channels, bias=False))
            self.lin_list.append(Linear(hidden_channels, output_channels, bias=False))

    def forward(self, x):
        x = self.lin_list[0](x)
        for lin in self.lin_list[1:]:
            x = x.relu()
            x = lin(x)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, random_seed=1234567, num_layers=2):
        super().__init__()
        torch.manual_seed(random_seed)
        # manifold coor
        self.rnn = torch.nn.LSTM(6, hidden_channels, num_layers=1, batch_first=True, bidirectional=True)
        # edge encoding
        self.mlp_combine = MLP(hidden_channels*2,hidden_channels,hidden_channels)
        
        # node coor
        self.mlp_coor = MLP(input_channels,hidden_channels,hidden_channels)
        
        # conv
        nn = MLP(hidden_channels,hidden_channels,output_channels)
        nn2 = MLP(hidden_channels*3,hidden_channels,hidden_channels)
        self.conv = GINEConv(nn=nn,nn2=nn2, eps=0.1)
        
    def forward(self, x, edge_index, edge_traj):
        # edge trajectory encoding
        output, (h_t, c_t) = self.rnn(edge_traj)
        traj_encoding = c_t.mean(dim=0)
        
        # node feature encoding
        coor_encoding = self.mlp_coor(x)
        
        out = self.conv(coor_encoding, edge_index, traj_encoding)
        
        # aggregate graph level embedding
        out = torch.sum(out, dim=0)  
        return out