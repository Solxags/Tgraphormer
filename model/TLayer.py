import torch
import torch.nn as nn

class GTU(nn.Module):
    def __init__(self, embed_dim,kernel_size, dilation):
        super(GTU, self).__init__()
        self.filter_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, kernel_size), dilation=dilation)
        self.gate_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, kernel_size), dilation=dilation)

    def forward(self, x):
        _filter = self.filter_conv(x)
        _filter = torch.tanh(_filter)
        _gate = self.gate_conv(x)
        _gate = torch.sigmoid(_gate)
        x_gtu = torch.mul(_filter, _gate)
        return x_gtu
    
class TLayer(nn.Module):
    def __init__(self, embed_dim, kernel_size,num_of_timesteps,dropout):
        super(TLayer, self).__init__()
        self.gtu1 = GTU( embed_dim,kernel_size,1)
        self.gtu2 = GTU( embed_dim, kernel_size,2)
        self.gtu4 = GTU( embed_dim, kernel_size,4)
        dim_all=3 * num_of_timesteps-(kernel_size-1)-(kernel_size-1)*2-(kernel_size-1)*4
        self.fcmy = nn.Sequential(
            nn.Linear(dim_all , num_of_timesteps),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # convolution along the time axis
        X = x.permute(0, 3, 2, 1)  # B,F,N,T
        x_gtu = []
        x_gtu.append(self.gtu1(X)) 
        x_gtu.append(self.gtu2(X))  
        x_gtu.append(self.gtu4(X))  
        time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,3T-(kernel_size-1)-(kernel_size-1)*2-(kernel_size-1)*4
        time_conv = self.fcmy(time_conv).transpose(1,3)#B,F,N,T
        return time_conv