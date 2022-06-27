import torch
import torch.nn.functional as F
from torch import nn

class POPI_hyper(nn.Module):
    
    def __init__(self, input_target=None, target_hidden_dim=4, target_hidden_size = 50, ray_hidden_dim=150):
        super().__init__()
        self.input_dim = input_target
        self.target_hidden_dim = target_hidden_dim
        self.target_hidden_size = target_hidden_size
        self.n_bounds = 2
        self.out_dim = 1
        self.ray_hidden_dim = ray_hidden_dim
        
        self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim)
        )
        
        self.fc_0_weights = nn.Linear(self.ray_hidden_dim, self.target_hidden_size * self.input_dim)
        self.fc_0_bias = nn.Linear(self.ray_hidden_dim, self.target_hidden_size)
        
        for i in range(1, self.target_hidden_dim + 1):
            setattr(self, f"fc_{i}_weights", nn.Linear(self.ray_hidden_dim, self.target_hidden_size * self.target_hidden_size))
            setattr(self, f"fc_{i}_bias", nn.Linear(self.ray_hidden_dim, self.target_hidden_size))
        
        for j in range(self.n_bounds):
            setattr(self, f"task_{j}_weights", nn.Linear(self.ray_hidden_dim, self.target_hidden_size * self.out_dim))
            setattr(self, f"task_{j}_bias", nn.Linear(self.ray_hidden_dim, self.out_dim))
    
    def shared_parameters(self):
        return list([p for n, p in self.named_parameters() if 'task' not in n])
    
    def forward(self, ray):
        x = self.ray_mlp(ray)
        
        out_dict = {}
        layer_types = ["fc", "task"]
        for i in layer_types:
            if i == "fc":
                n_layers = self.target_hidden_dim + 1
            elif i == "task":
                n_layers = self.n_bounds
            
            for j in range(n_layers):
                out_dict[f"{i}{j}.weights"] = getattr(self, f"{i}_{j}_weights")(x)#.reshape(self.target_hidden_dim, self.input_dim)
                out_dict[f"{i}{j}.bias"] = getattr(self, f"{i}_{j}_bias")(x).flatten()
        
        return out_dict


class POPI_target(nn.Module):
    
    def __init__(self, input_target=6, target_hidden_dim=4, target_hidden_size = 100):
        super().__init__()
        self.input_dim = input_target
        self.target_hidden_dim = target_hidden_dim
        self.target_hidden_size = target_hidden_size
        self.n_bounds = 2
        self.out_dim = 1
        
    def forward(self, x, weights):
        
        x = F.linear(
            x,
            weight=weights["fc0.weights"].reshape(self.target_hidden_size, x.shape[-1]),
            bias=weights["fc0.bias"]
        )
        x = F.elu(x)
        
        
        for i in range(1,self.target_hidden_dim + 1):
            x = F.linear(
                x,
                weight=weights[f'fc{i}.weights'].reshape(self.target_hidden_size, self.target_hidden_size),
                bias=weights[f'fc{i}.bias']
            )
            x = F.elu(x)
        
        pred_int = []
        for j in range(self.n_bounds):
            pred_int.append(
                F.linear(
                    x, weight=weights[f'task{j}.weights'].reshape(self.out_dim, self.target_hidden_size),
                    bias=weights[f'task{j}.bias']
                )
            )
        
        return pred_int