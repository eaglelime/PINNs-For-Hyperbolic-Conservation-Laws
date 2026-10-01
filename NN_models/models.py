from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np

class FlexibleNet(nn.Module):
    def __init__(self, width, depth, activation, input, output):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(input,width))
        layers.append(activation())
        for i in range(depth):
            layers.append(nn.Linear(width,width))
            layers.append(activation())
        layers.append(nn.Linear(width, output))
        self.net = nn.Sequential(*layers)
        torch.set_default_dtype(torch.float64)
        self.activation = activation
        self.apply(self._init_weights)
        self.double()

    #inits weights for every layer.
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            if self.activation == nn.Tanh:
                #Xavier is commonly used for tanh.
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.net(x)
        return logits


class FlexibleNetPeriodicX(nn.Module):
    def __init__(self, width, depth, activation, input, output, x_size):
        super().__init__()
        self.x_size = x_size
        self.flatten = nn.Flatten()
        layers = []
        layers.append(nn.Linear(input,width))
        layers.append(activation())
        for i in range(depth):
            layers.append(nn.Linear(width,width))
            layers.append(activation())
        layers.append(nn.Linear(width, output))
        self.net = nn.Sequential(*layers)
        torch.set_default_dtype(torch.float64)
        self.activation = activation
        self.apply(self._init_weights)
        self.double()

    #inits weights for every layer.
    def _init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            if self.activation == nn.Tanh:
                #Xavier is commonly used for tanh.
                nn.init.xavier_uniform_(layer.weight)
            else:
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        # 1. Transform x to be inherently periodic
        x_cos = torch.cos(2 * torch.pi * x[:, 0:1] / self.x_size)
        x_sin = torch.sin(2 * torch.pi * x[:, 0:1] / self.x_size)

        # 2. Concatenate periodic spatial features with time
        nn_input = torch.cat([x_cos, x_sin, x[:, 1:2]], dim=-1)
        
        # 3. Pass through the network
        return self.net(nn_input)
        #x = self.flatten(x)
        #logits = self.net(x)
        #return logits

class FlexibleWeakEulerNet(FlexibleNet):

    def forward(self, x):
        logits = super().forward(x)
        out = torch.stack([torch.nn.functional.softplus(logits[:,0]),
                          logits[:,1],
                          torch.nn.functional.softplus(logits[:,2])],1)
        return out

#torch.nn.functional.softplus(

class NetTrainer():
    def __init__(self, losses, optimizers, schedulers, network, max_clip, print_fn):
        self.losses = losses
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.network = network
        self.max_clip = max_clip
        self.print_fn = print_fn
        self.network.train()

    def train(self, epoch, print_losses = False):
        for opt in self.optimizers:
            opt.zero_grad()
        if(print_losses):
            print("--------------MAIN---------------")
        loss = sum([l.get_loss(epoch, print_losses) for l in self.losses])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_clip)
        for opt, sched in zip(self.optimizers, self.schedulers):
            opt.step()
            opt.zero_grad()
            sched.step()
        return loss.detach()

class AdvNetTrainer():
    def __init__(self,training_per_epoch, entropy_loss, losses, init_training, network, max_clip, print_fn, reset_frequency, offset = 0):
        self.entropy_loss = entropy_loss
        self.losses = losses
        self.optimizers, self.schedulers = init_training(network)
        self.init_training = init_training
        self.network = network
        self.max_clip = max_clip
        self.print_fn = print_fn
        self.network.train()
        self.training_per_epoch = training_per_epoch
        self.reset_frequency = reset_frequency
        self.offset = offset

    def train(self, epoch, print_losses = False):
        for opt in self.optimizers:
            opt.zero_grad()
        if (epoch+self.offset) % self.reset_frequency == self.reset_frequency - 1:
            self.optimizers, self.schedulers = self.init_training(self.network)
        if(print_losses):
            print("-------------ADV----------------")
        for i in range(self.training_per_epoch):
            loss = sum([l.get_loss(epoch, print_losses) for l in self.losses])-self.entropy_loss.get_loss(epoch, print_losses and (i == 0 or i == self.training_per_epoch - 1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_clip)
            for opt, sched in zip(self.optimizers, self.schedulers):
                opt.step()
                opt.zero_grad()
                sched.step()

#TODO: maybe move print functions to burgers.py or smt??
def print_test_network_output(test_function, save_path, boundaries, suffix = ""):
    x = torch.linspace(boundaries.x_lb, boundaries.x_ub, 500)
    t = torch.linspace(boundaries.t_lb, boundaries.t_ub, 500)
    mesh_x,mesh_t = torch.meshgrid(x.squeeze(),t.squeeze(),indexing="ij")
    X = torch.stack((mesh_x, mesh_t), dim=2)
    X = torch.flatten(X, end_dim=1)

    X = X#.to(device)

    with torch.no_grad():
        u_pred = test_function(X).squeeze()
        #u_pred = u_pred / torch.max(u_pred)
    if u_pred[1].numel() > 1:
        u_pred = u_pred[:,0]
    u_pred = u_pred.cpu().numpy()
    u_pred = u_pred.reshape(x.numel(), t.numel())
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    min = np.min(u_pred)
    max = np.max(u_pred)
    cp = plt.pcolormesh(mesh_t.cpu(), mesh_x.cpu(), u_pred, cmap='rainbow', shading='nearest', clim = [min,max])
    plt.colorbar(cp).set_label('$u(x, t)$', rotation=0, labelpad=15)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Test net')
    plt.savefig(save_path/f"test_net{suffix}.jpg")
    plt.close()

    #print(f"Saved output to {save_path}")


def print_model_output(test_data, network, output_path, suffix = ""):
    x, t, usol = test_data.x,test_data.t,test_data.usol.cpu().numpy()

    mesh_x,mesh_t = torch.meshgrid(x.squeeze(),t.squeeze(),indexing="ij")
    X = torch.stack((mesh_x, mesh_t), dim=2)
    X = torch.flatten(X, end_dim=1)
    X = X.to(torch.get_default_device())

    with torch.no_grad():
        u_pred = network(X).cpu().numpy()

    u_pred = u_pred.reshape(x.numel(), t.numel())
    u_diff = np.subtract(usol,u_pred)
    save_1d_output_graphs(output_path, usol, u_pred, mesh_x.cpu(), mesh_t.cpu(), suffix)

def print_models_output(test_data, networks, output_path, suffix = ""):
    x, t, usol = test_data.x,test_data.t,test_data.usol.cpu().numpy()
    slices = len(networks)

    mesh_x, mesh_t = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    t_chunks = torch.chunk(t.squeeze(), slices)
    u_pred_chunks = []
    for net, t_chunk in zip(networks, t_chunks):
        mesh_x_i, mesh_t_i = torch.meshgrid(x.squeeze(), t_chunk, indexing="ij")
        X_i = torch.stack((mesh_x_i, mesh_t_i), dim=2).flatten(end_dim=1)
        X_i = X_i.to(torch.get_default_device())
        with torch.no_grad():
            u_chunk = net(X_i).cpu().numpy().reshape(x.numel(), t_chunk.numel())
        u_pred_chunks.append(u_chunk)
    u_pred = np.concatenate(u_pred_chunks, axis=1)
    save_1d_output_graphs(output_path, usol, u_pred, mesh_x.cpu(), mesh_t.cpu(), suffix)

def save_1d_output_graphs(save_path, usol, u_pred, mesh_x, mesh_t, suffix):
    save_path.mkdir(parents=True, exist_ok=True)

    u_diff = np.subtract(usol,u_pred)

    v_min = np.min([np.min(usol),np.min(u_pred),np.min(u_diff)])
    v_max = np.max([np.max(usol),np.max(u_pred),np.max(u_diff)])
    def plot_and_save(data, title, filename):
        plt.figure(figsize=(10, 6))
        cp = plt.pcolormesh(mesh_t, mesh_x, data, cmap='rainbow', 
                            shading='nearest', vmin=v_min, vmax=v_max)
        plt.colorbar(cp).set_label('$u(x, t)$', rotation=0, labelpad=15)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.title(title)
        #plt.tight_layout()
        plt.savefig(save_path / filename, format='jpg')
        plt.close()

    plot_and_save(u_diff, 'Model Error Residue', "output_diff.jpg")
    plot_and_save(u_pred, 'Model Prediction', f"output{suffix}.jpg")
    plot_and_save(usol, 'Test Data', "test_data.jpg")

    #print(f"Saved output to {save_path}")