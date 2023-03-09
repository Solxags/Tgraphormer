from .layer_connector import FeedForward
from .SpatialAttention import SpatialAttention
from .TLayer import TLayer
from .TGraphormer import TGraphormer_layer,TGraphormer

import torch.nn as nn

def make_model(in_dim,embed_dim,num_of_timesteps,num_heads,d_ff,kernel_size,N,max_degree,end_dim,out_dim,dropout):
    SAT=SpatialAttention(embed_dim,embed_dim,num_of_timesteps,num_heads,dropout)
    FFN=FeedForward(embed_dim,d_ff,dropout)
    TGC=TLayer(embed_dim,kernel_size,num_of_timesteps,dropout)
    layer=TGraphormer_layer(SAT,FFN,TGC,embed_dim,dropout)
    model=TGraphormer(in_dim,layer,N,max_degree,embed_dim,end_dim,out_dim)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model



