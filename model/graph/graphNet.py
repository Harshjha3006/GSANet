import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, EdgeConv, SAGEConv, BatchNorm
import torch


class GraphNet(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,final_dim,dropout):
        super().__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.final_dim = final_dim

        # Initial Layer to convert to hidden_dim
        self.layer0 = Linear(input_dim,hidden_dim1,weight_initializer="kaiming_uniform")
        self.batch0 = BatchNorm(hidden_dim1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # EdgeConv Layers 
        # For forward edges
        self.layer11 = EdgeConv(nn.Sequential(Linear(2 * hidden_dim1, hidden_dim1,weight_initializer="kaiming_uniform"),nn.ReLU(),
                                              Linear(hidden_dim1,hidden_dim1,weight_initializer="kaiming_uniform")))
        self.batch11 = BatchNorm(hidden_dim1)
        # For backward edges
        self.layer12 = EdgeConv(nn.Sequential(Linear(2 * hidden_dim1, hidden_dim1,weight_initializer="kaiming_uniform"),nn.ReLU(),
                                              Linear(hidden_dim1,hidden_dim1,weight_initializer="kaiming_uniform")))
        self.batch12 = BatchNorm(hidden_dim1)
        # For undirected edges
        self.layer13 = EdgeConv(nn.Sequential(Linear(2 * hidden_dim1, hidden_dim1,weight_initializer="kaiming_uniform"),nn.ReLU(),
                                              Linear(hidden_dim1,hidden_dim1,weight_initializer="kaiming_uniform")))
        self.batch13 = BatchNorm(hidden_dim1)


        # Common SageConv layer
        self.layer2 = SAGEConv(hidden_dim1,hidden_dim2,project=True)
        self.batch2 = BatchNorm(hidden_dim2)


        # SageConv layers
        self.layer31 = SAGEConv(hidden_dim2,final_dim,project=True)
        self.layer32 = SAGEConv(hidden_dim2,final_dim,project=True)
        self.layer33 = SAGEConv(hidden_dim2,final_dim,project=True)
        

    def forward(self,x,edge_index,edge_attr):
        # x's shape -> (b,num_nodes,feature_dim)
        # edge_index -> (2,num_edges)
        # edge_attr -> (1,num_edges)
        # Applying initial layers to x
        x_old = x
        x = self.layer0(x)
        x = torch.stack([self.batch0(batch) for batch in x],dim = 0)


        edge_index_f = edge_index[:,edge_attr >= 0]
        edge_index_b = edge_index[:,edge_attr < 0]

        # Applying edgeConv to forward edges
        x1 = self.layer11(x,edge_index_f)
        x1 = torch.stack([self.batch11(batch) for batch in x1],dim = 0)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Applying sageConv to forward edges
        x1 = self.layer2(x1,edge_index_f)
        x1 = torch.stack([self.batch2(batch) for batch in x1],dim = 0)

        # Applying edgeConv to backward edges
        x2 = self.layer12(x,edge_index_b)
        x2 = torch.stack([self.batch12(batch) for batch in x2],dim = 0)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        
        # Applying sageConv to backward edges
        x2 = self.layer2(x2,edge_index_b)
        x2 = torch.stack([self.batch2(batch) for batch in x2],dim = 0)


        # Applying edgeConv to undirected edges
        x3 = self.layer13(x,edge_index)
        x3 = torch.stack([self.batch13(batch) for batch in x3],dim = 0)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        # Applying sageConv to undirected edges
        x3 = self.layer2(x3,edge_index)
        x3 = torch.stack([self.batch2(batch) for batch in x3],dim = 0)

        # Applying 2nd sageConv to 3 types of edges
        x1 = self.layer31(x1,edge_index_f)
        x2 = self.layer32(x2,edge_index_b)
        x3 = self.layer33(x3,edge_index)
        # output's shape is (b,num_nodes,final_dim = 1)
        out = x1 + x2 + x3
        out = F.softmax(out,dim = 1)
        # result's shape is (b,num_nodes,1024)
        result = out * x_old
        return result
        




        
                