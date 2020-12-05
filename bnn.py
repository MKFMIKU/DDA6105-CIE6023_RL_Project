import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.io import savemat, loadmat


class BNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(23, 200, bias=True)
        self.fc2 = nn.Linear(200, 200, bias=True)
        self.fc3 = nn.Linear(200, 200, bias=True)
        self.fc4 = nn.Linear(200, 200, bias=True)
        self.var = nn.Linear(200, 18, bias=True)

        self.two_var = nn.Linear(200, 18, bias=True)

    def silu(self, x):
        return x * F.sigmoid(x)
    
    def forward(self, x):
        x = self.silu(self.fc1(x))
        x = self.silu(self.fc2(x))
        x = self.silu(self.fc3(x))
        var = self.two_var(x)
        x = self.silu(self.fc4(x))
        x = self.var(x)
        return x, var


class EnsembledBNN(nn.Module):
    def __init__(self, ensemble_size=7):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = [BNN() for i in range(ensemble_size)]

    # ../ray_mopo/halfcheetah/halfcheetah_medium_replay_101e3/seed:3316_2020-11-12_08-25-1462dxwsw1/models/BNN_0.mat
    def load_weight(self, weight_path):
        params_dict = loadmat(weight_path)

        for i in range(self.ensemble_size):
            self.models[i].fc1.weight.data = torch.from_numpy(params_dict['2'][i].transpose(1, 0)).double().cuda()
            self.models[i].fc1.bias.data = torch.from_numpy(params_dict['3'][i][0]).double().cuda()

            self.models[i].fc2.weight.data = torch.from_numpy(params_dict['4'][i].transpose(1, 0)).double().cuda()
            self.models[i].fc2.bias.data = torch.from_numpy(params_dict['5'][i][0]).double().cuda()
    
            self.models[i].fc3.weight.data = torch.from_numpy(params_dict['6'][i].transpose(1, 0)).double().cuda()
            self.models[i].fc3.bias.data = torch.from_numpy(params_dict['7'][i][0]).double().cuda()

            self.models[i].fc4.weight.data = torch.from_numpy(params_dict['8'][i].transpose(1, 0)).double().cuda()
            self.models[i].fc4.bias.data = torch.from_numpy(params_dict['9'][i][0]).double().cuda()

            self.models[i].var.weight.data = torch.from_numpy(params_dict['10'][i].transpose(1, 0)).double().cuda()
            self.models[i].var.bias.data = torch.from_numpy(params_dict['11'][i][0]).double().cuda()

            self.models[i].two_var.weight.data = torch.from_numpy(params_dict['12'][i].transpose(1, 0)).double().cuda()
            self.models[i].two_var.bias.data = torch.from_numpy(params_dict['13'][i][0]).double().cuda()


    def predict(self, inputs, factored=True):
        inputs = torch.from_numpy(inputs).double().cuda()
        mean_outputs = []
        logvar_outputs = []

        with torch.no_grad():
            for m in self.models:
                mean, logvar = m(inputs)
                mean_outputs.append(mean)
                logvar_outputs.append(logvar)
        mean = torch.stack(mean_outputs, 0)
        logvar = torch.stack(logvar_outputs, 0)
        return mean.cpu().numpy(), logvar.cpu().numpy()