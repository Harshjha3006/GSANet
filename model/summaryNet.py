import torch
import torch.nn as nn 
import numpy as np
import torch.nn.functional as F

from torch_geometric.data import Data


from model.mha.attention import MultiHeadAttention
from model.graph.graphNet import GraphNet

def build_edges(num_segs,threshold):

    edge_source,edge_dest,edge_attr = [],[],[]

    for i in range(num_segs):
        for j in range(num_segs):
            if abs(i - j) <= threshold:
                edge_source.append(i)
                edge_dest.append(j)
                edge_attr.append(np.sign(i - j))

    return edge_source,edge_dest,edge_attr


def build_graph(block_outputs,threshold):
    edge_source,edge_dest,edge_attr = build_edges(block_outputs.shape[1],threshold)
    return Data(x = block_outputs,
                edge_index = torch.tensor(np.array([edge_source,edge_dest],dtype = np.int64),dtype = torch.long),
                edge_attr = torch.tensor(edge_attr,dtype = torch.long))
    
class SummaryNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.num_blocks = config.num_segments
        self.threshold = config.threshold
        self.mhaList = nn.ModuleList([MultiHeadAttention(config.embed_dim,config.num_heads) for _ in range(config.num_layers)])
        self.graphnet = GraphNet(config.embed_dim,config.hidden_dim1,config.hidden_dim2,config.final_dim,config.dropout_rate)
        self.linear2 = nn.Linear(config.embed_dim,2)
        self.linear = nn.Linear(config.embed_dim,config.embed_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm(config.embed_dim)
        self.sig = nn.Sigmoid()
        self.segment_weight = nn.Parameter(torch.tensor(1.0,requires_grad = True))
        self.video_weight = nn.Parameter(torch.tensor(1.0,requires_grad=True))
        self.frame_weight = nn.Parameter(torch.tensor(1.0,requires_grad=True))
        self.layer_norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self,x,y):
        batch_size,seq_len,embed_dim = x.shape
        block_size = seq_len // self.num_blocks
        new_seq_len = block_size * self.num_blocks
        x = x[:,:new_seq_len,:]
        y = y[:,:new_seq_len,:]

        # seq_len = num_blocks * bsize
        x = torch.reshape(x,(batch_size,self.num_blocks,block_size,embed_dim))

        block_outputs = []
        for i in range(self.num_blocks):
            # block -> (b,bsize,1024)
            block = x[:,i,:,:]
            # (bo -> (b,1024))
            for mha in self.mhaList:
              block_output,block = mha(block)
            block_outputs.append(block_output)
        # (b,num_blocks,1024)
        block_outputs = torch.stack(block_outputs,dim = 1)

        with torch.no_grad():
            graph_data = build_graph(block_outputs,self.threshold)
        
        graph_data = graph_data.to(x.device)
        # (b,num_blocks,1024)
        graph_output = self.graphnet(graph_data.x,graph_data.edge_index,graph_data.edge_attr)
        # (b,num_blocks,1,1024) -> (b,num_blocks,bsize,1024)
        graph_output = torch.unsqueeze(graph_output,dim = 2).expand(batch_size,self.num_blocks,block_size,embed_dim)

        # (b,num_blocks,bsize,1024)
        block_outputs = torch.unsqueeze(block_outputs,dim = 2).expand(batch_size,self.num_blocks,block_size,embed_dim)

        # x -> Represents a single frame's content
        # block_outputs -> Represents a frame's segment content
        # graph_output -> Represents that segment's importance in the context of the whole video 
        out = self.frame_weight * x + self.segment_weight * block_outputs + self.video_weight * graph_output
        out = torch.reshape(out,(batch_size,self.num_blocks * block_size,embed_dim))
        out = self.layer_norm2(out)

        # Regressor Network 
        # 1024 -> 1024
        out = self.linear(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.layer_norm(out)
        # 1024 -> 2
        out = self.linear2(out)
        # out = self.sig(out)
        return out,y


        

        











        



        