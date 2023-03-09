import torch.nn as nn
import torch


#在加上attn_bias的情况下计算各节点的attention
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5


        self.k_proj =nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # self.reset_parameters()

    def forward(
        self,
        query,
        attn_bias
    ) :
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        #query=[b*n*f]
        (bsz,length,n_feature)=query.shape
        q = self.q_proj(query)
        k = self.k_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(bsz, -1,self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        k = (
            k.contiguous()
            .view(bsz, -1,self.num_heads, self.head_dim)
            .transpose(1, 2)
        )


        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        #attn_bias=[b*num_heads*n*n]
        if attn_bias is not None:
            attn_weights += attn_bias


        attn_probs = attn_weights.softmax(dim=-1)

        #attn_probs=[b*n_heads*n*n]

        return attn_probs

#计算地理信息上的attention
class SpatialAttention(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        src_len,
        num_heads,
        dropout=0.1,
        bias=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.src_len=src_len

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.attention_layer=MultiheadAttention(embed_dim,num_heads,bias)
        self.enter=nn.Conv2d(src_len, 1, kernel_size=(1, 1))
        self.v_con=nn.Conv2d(in_dim,self.head_dim,kernel_size=(1,1))
        self.dropout = nn.Dropout(dropout)
        # self.reset_parameters()
    def forward(
        self,
        input,
        in_degree_emb,
        out_degree_emb,
        shortest_path_emb,
        shortest_hop_emb
    ) : 
        bsz=input.shape[0]
        #input=[b*src_len*n*embed_dim]
        x=self.enter(input).squeeze(1)
        #x=[b*n*embed_dim],in_degree_emb=out_degree_emb=[n*embed_dim]
        x=x+in_degree_emb+out_degree_emb
        #attn_prob=[b*head_dim*n*n]
        shortest_path_emb=shortest_path_emb.repeat(bsz,self.num_heads,1,1)
        shortest_hop_emb=shortest_hop_emb.repeat(bsz,self.num_heads,1,1)
        attn_prob=self.attention_layer(x,shortest_path_emb+shortest_hop_emb)

        #v=[b*n*head_dim*src_len]
        v=self.v_con(input.transpose(1,3)).transpose(1,2)
        result=torch.einsum('bnuv,bvft->bnuft', [attn_prob,v])
        #result=[b*num_heads*n*head_dim*src_len]
        result=(
            result.transpose(1, 4)
            .contiguous()
            .view(bsz,self.src_len,-1,self.num_heads * self.head_dim)
        )
        #result=[b*src_len*n*embed_dim]
        return self.dropout(result)
