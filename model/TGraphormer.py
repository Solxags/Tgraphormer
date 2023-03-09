import torch.nn as nn
import torch
from .layer_connector import LayerNorm,SublayerConnection,clones

class TGraphormer(nn.Module):
    def __init__(self, in_dim,layer, N,max_degree,embed_dim,end_dim,out_dim):
        super(TGraphormer, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(embed_dim)
        self.in_degree_encoder = nn.Embedding(max_degree+1, embed_dim)
        self.out_degree_encoder = nn.Embedding(max_degree+1, embed_dim)
        self.shortest_hop_encoder= nn.Embedding(max_degree, 1)
        self.enter=nn.Linear(in_dim, embed_dim)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Linear(embed_dim, end_dim),
            nn.ReLU(),
            nn.Linear(end_dim, out_dim)
        )

    def forward(self, x,in_degree,out_degree,shortest_path_result,shortest_hop_result):
        #x=[b*src_len*n*f] in_degree=out_degree=[n] shortest_path_result=[n*n] short_hop_result=[n*n] 
        in_degree_emb=self.in_degree_encoder(in_degree)
        out_degree_emb=self.out_degree_encoder(out_degree)
        shortest_hop_emb=self.shortest_hop_encoder(shortest_hop_result).permute(2,0,1)
        shortest_path_emb=shortest_path_result
        x=self.enter(x)
        #x=[b*src_len*n*f]
        for layer in self.layers:
            x = layer(x,in_degree_emb,out_degree_emb,shortest_path_emb,shortest_hop_emb)
        return self.out(self.norm(x))

class TGraphormer_layer(nn.Module):
    def __init__(self, SAT,feed_forward,TGC,embded_dim,dropout):
        super(TGraphormer_layer, self).__init__()
        self.SpatialAttention=SAT
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(embded_dim, dropout), 3)
        self.TLayer=TGC
    def forward(self,
                x,
                in_degree_emb,
                out_degree_emb,
                shortest_path_emb,
                shortest_hop_emb
                ):
        x=self.sublayer[0](x,lambda x:self.SpatialAttention(x,in_degree_emb,out_degree_emb,shortest_path_emb,shortest_hop_emb))
        x=self.sublayer[1](x,self.feed_forward)
        return self.sublayer[2](x,lambda x:self.TLayer(x))
