import torch.nn as nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.networks import Mlp
from rlkit.pythonplusplus import identity
import torch
import copy

class FedFormer(PyTorchModule):
    def __init__(self, hidden_sizes, output_size, input_size, transformer_layer_config, num_layers, agent_index, num_agents=5,
                 from_saved=0,
                 saved_id=None,
                 init_w=3e-3,
                 hidden_activation=F.relu, output_activation=identity, hidden_init=ptu.fanin_init, b_init_value=0.,
                 layer_norm=False, layer_norm_kwargs=None):
        super().__init__()
        models = []
        if from_saved != 0 and saved_id != None:
            models += [torch.load(f'networks/{saved_id}/encoder-{i}.pt') for i in range(from_saved)]
            agent_index = agent_index + from_saved
         
        models += [Mlp(hidden_sizes,
                                            hidden_sizes[-1],
                                            input_size,
                                            init_w,
                                            hidden_activation,
                                            output_activation,
                                            hidden_init,
                                            b_init_value,
                                            layer_norm,
                                            layer_norm_kwargs) for _ in range(num_agents)]
        

        self.encoders = nn.ModuleList(models)
        num_agents = num_agents + from_saved

        self.positional_encoding = nn.Embedding(num_agents + 1, hidden_sizes[-1])
        transformer_encoder_layer = nn.TransformerEncoderLayer(**transformer_layer_config)
        self.transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers)

        self.decoder = Mlp(hidden_sizes,
                                           output_size,
                                           hidden_sizes[-1] * 2,
                                           init_w,
                                           hidden_activation,
                                           output_activation,
                                           hidden_init,
                                           b_init_value,
                                           layer_norm,
                                           layer_norm_kwargs)
        
        
        self.cls = nn.Embedding(2, hidden_sizes[-1])
        
        
        self.agent_index = agent_index
        self.num_agents = num_agents
        
        for encoder in self.encoders:
            if type(encoder.requires_grad_) != bool:
                encoder.requires_grad_(False)
        self.encoders[self.agent_index].requires_grad_(True)
           

    def forward(self, obs, actions):
        x = torch.cat((obs, actions), dim=1)  
   
        encodings = [torch.tile(self.cls(torch.tensor(1, dtype=torch.int).to('cuda:0')), (x.shape[0], 1))]
        encodings += [net(x) for net in self.encoders]

        encodings = torch.stack(encodings, dim=0)
        indices = torch.arange(0, self.num_agents + 1, dtype=torch.int).to('cuda:0') # I hope this works!
        encodings_transformer = encodings + torch.tile(self.positional_encoding(indices).unsqueeze(1), (1, x.shape[0], 1))

        transformed = self.transformer(encodings_transformer)  
        
        decoded = self.decoder(torch.cat((transformed[0], encodings[self.agent_index + 1]), dim=1))  # we use other networks to weight entropy perhaps

        return decoded  # return decoded for the current agent

    def get_local_encoder(self):
        net = copy.deepcopy(self.encoders[self.agent_index])
        net.requires_grad_ = False
        return net
    
    def get_encoders(self):
        return self.encoders
    
    def fuse(self, other):
        index = other.agent_index
        self.encoders[index] = other.get_local_encoder()
        other.encoders[self.agent_index] = self.get_local_encoder()
        

