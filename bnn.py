import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembledBNN(nn.Module):
    def __init__(self, ensemble_size = 7):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = [BNN() for i in range(ensemble_size)]

    # ../ray_mopo/halfcheetah/halfcheetah_medium_replay_101e3/seed:3316_2020-11-12_08-25-1462dxwsw1/models/BNN_0.mat
    def load_weight(self, weight_path):
        params_dict = loadmat(weight_path)

        for i in range(ensemble_size):
            self.models[i].fc1.weight.data = torch.from_numpy(params_dict['2'][i].transpose(1, 0))
            self.models[i].fc1.bias.data = torch.from_numpy(params_dict['3'][i][0])

            self.models[i].fc2.weight.data = torch.from_numpy(params_dict['4'][i].transpose(1, 0))
            self.models[i].fc2.bias.data = torch.from_numpy(params_dict['5'][i][0])
    
            self.models[i].fc3.weight.data = torch.from_numpy(params_dict['6'][i].transpose(1, 0))
            self.models[i].fc3.bias.data = torch.from_numpy(params_dict['7'][i][0])

            self.models[i].fc4.weight.data = torch.from_numpy(params_dict['8'][i].transpose(1, 0))
            self.models[i].fc4.bias.data = torch.from_numpy(params_dict['9'][i][0])

            self.models[i].var.weight.data = torch.from_numpy(params_dict['10'][i].transpose(1, 0))
            self.models[i].var.bias.data = torch.from_numpy(params_dict['11'][i][0])

    def predict(self, inputs, factored=True):
        outputs = []
        for i in range(inputs.shape[0]):
            output = self.models[i](inputs[i])
            outputs.append(output)
        return torch.stack(output, 0)