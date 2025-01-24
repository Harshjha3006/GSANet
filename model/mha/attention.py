import torch
import torch.nn as nn
import torch.nn.functional as F
import math
    
def get_pos_encoding(x):
        # let's assume , x -> (1,10,512)
        # pos -> (10,1)
        _,seq_len,embed_dim = x.shape
        pos = torch.arange(seq_len).float().reshape(seq_len,1)
        # even_idx -> (1,256)
        even_idx = torch.arange(0,embed_dim,2).float()
        # even denominator = odd denominator
        # denominator -> (1,256)
        denominator = torch.pow(10000,even_idx/embed_dim)
        # even_pe -> (10,256)
        even_pe = torch.sin(pos/denominator)
        # odd_pe -> (10,256)
        odd_pe = torch.cos(pos/denominator)
        # pe -> (10,256,2)
        pe = torch.stack([even_pe,odd_pe],dim = 2)
        # pe -> (10,512)
        pe = torch.flatten(pe,start_dim=1,end_dim = 2)
        # pe -> (1,10,512)
        pe = torch.unsqueeze(pe,0)
        pe = pe.to(x.device)

        return x + pe


class MultiHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim_per_head = embed_dim // num_heads
        self.qkvLayer = nn.Linear(embed_dim,embed_dim * 3)
        self.linear = nn.Linear(embed_dim,embed_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(embed_dim,1)

    def forward(self,x):
        # let's assume seq_len = 10, embed_dim = 512,num_heads = 8, embed_dim_per_head = 64
        # seq_len is segment size
        batch_size,seq_len,embed_dim = x.size()
        # Adds positional encoding to input
        x = get_pos_encoding(x)
        # (1,10,512) -> (1,10,1536)
        qkv = self.qkvLayer(x)
        # (1,10,1536) -> (1,10,8,192)
        qkv = torch.reshape(qkv,(batch_size,seq_len,self.num_heads,3 * self.embed_dim_per_head))
        # (1,10,8,192) -> (1,8,10,192) each head has a sequence of 10 elements with head_dim * 3
        qkv = torch.permute(qkv,(0,2,1,3))
        # (1,8,10,192) -> 3 * (1,8,10,64)
        q,k,v = torch.chunk(qkv,3,dim = -1)

        # Self Attention
        # attention -> (1,8,10,10)
        attention = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(embed_dim)
        attention = F.softmax(attention,dim = -1)
        # values -> (1,8,10,64)
        values = torch.matmul(attention,v)
        # (1,8,10,64) -> (1,10,8,64)
        values = torch.permute(values,(0,2,1,3))
        # (1,10,8,64) -> (1,10,512)
        values = torch.reshape(values,(batch_size,seq_len,self.num_heads * self.embed_dim_per_head))
        # (1,10,512) -> (1,10,512)
        out = self.linear(values)
        out = self.relu(out)
        
        # Now output has context aware feature vectors of frames in the sequence 
        # (1,10,512) -> (1,10,1)
        weights = self.linear2(out)
        # Normalising the weights
        weights = F.softmax(weights,dim = 1)

        # (1,10,512)
        weighted_embeddings = weights * out
        # result -> (1,512)
        # generating the representation of 1 segment by calculating the weighted average of all frames
        result = torch.sum(weighted_embeddings,dim = 1)
        return result,out