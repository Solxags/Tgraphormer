import torch.nn as nn
import torch
import copy

class LayerNorm(nn.Module):
    '''
    构造一个'层归一化'模块
    '''

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # -1 表示计算沿着张量的最后一维度的平均值。keepdim 是一个布尔值，用于指定是否保留维度。
        # 如果将 keepdim 设置为 True，则输出张量的形状将与输入张量的形状相同，只是最后一维的大小为 1。
        # 如果将 keepdim 设置为 False，则输出张量的形状将不包括最后一维。
        mean = x.mean(-1, keepdim=True) 
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class FeedForward(nn.Module):
    '''
    实现一个FFN模型
    '''

    def __init__(self, embed_dim, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(embed_dim, d_ff)
        self.w_2 = nn.Linear(d_ff, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
class SublayerConnection(nn.Module):
    """
    残差链接
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
def clones(module, N):
    '''
    克隆N层一模一样的
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])