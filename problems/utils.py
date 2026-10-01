import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
torch.set_default_dtype(torch.float64)
torch.set_default_device(device)
import numpy as np

class TrainingParams():
    def __init__(self, batch_size_ic = 64, batch_size_bc = 64, batch_size_pde = 128, epochs = 50,start_lr = 5e-2, end_lr = 5e-2):
        self.batch_size_ic = batch_size_ic
        self.batch_size_bc = batch_size_bc
        self.batch_size_pde = batch_size_pde
        self.epochs = epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
        

class Boundaries():
    def __init__(self,x_lb,x_ub,t_lb,t_ub,x_size,t_size):
        self.x_lb, self.x_ub, self.x_size = x_lb, x_ub, x_size
        self.t_lb, self.t_ub, self.t_size = t_lb, t_ub, t_size

class TestDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.x, self.t, self.usol = torch.from_numpy(data['x'].T), torch.from_numpy(data['t'].T), torch.from_numpy(data['usol'])


    def __len__(self):
        return self.usol.shape[0] * self.usol.shape[1]

    def __getitem__(self, idx):
        x_i = idx % self.x.numel()
        t_i = idx //self.x.numel()
        return torch.cat([self.x[x_i],self.t[t_i]]), self.usol[x_i,t_i]



def plot_losses(losses, path):
    epochs = [e for e, _ in losses]
    vals = [float(loss) for _, loss in losses]
    fig, ax = plt.subplots()
    ax.plot(epochs, vals)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    fig.savefig(path)
    plt.close(fig)


def print_grad_norm(parameters):
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Total grad norm: {total_norm:.6f}")


def get_legendre(n, x):
    n = int(n)
    if n == 0:
        return torch.ones_like(x)
    p_prev2 = torch.ones_like(x)
    p_prev1 = x.clone()
    for k in range(2, n + 1):
        p_curr = ((2 * k - 1) * x * p_prev1 - (k - 1) * p_prev2) / k
        p_prev2 = p_prev1
        p_prev1 = p_curr
    return p_prev1

def get_legendre_scaled(n,X, boundaries):
    x = X[:,0]
    t = X[:,1]
    x_scaled = 2*(x-boundaries.x_lb)/(boundaries.x_ub-boundaries.x_lb)-1
    t_scaled = 2*(t-boundaries.t_lb)/(boundaries.t_ub-boundaries.t_lb)-1
    return get_legendre(n,x_scaled)*(1-x_scaled**2)*get_legendre(n,t_scaled)*(1-t_scaled**2)

def random_legendre(X, boundaries):
    n = torch.randint(0,20, (1,)).to(device)
    x = X[:,0]
    t = X[:,1]
    x_scaled = 2*(x-boundaries.x_lb)/(boundaries.x_ub-boundaries.x_lb)-1
    t_scaled = 2*(t-boundaries.t_lb)/(boundaries.t_ub-boundaries.t_lb)-1
    return get_legendre(n,x_scaled*torch.rand(1).to(device))*(1-x_scaled**2)*get_legendre(n,t_scaled*torch.rand(1).to(device))*(1-t_scaled**2)

def random_legendre2(X, boundaries):
    n = torch.randint(0,20, (1,)).to(device)
    x = X[:,0]
    t = X[:,1]
    x_scaled = 2*(x-boundaries.x_lb)/(boundaries.x_ub-boundaries.x_lb)-1
    t_scaled = 2*(t-boundaries.t_lb)/(boundaries.t_ub-boundaries.t_lb)-1
    return get_legendre(n,x_scaled*torch.rand(1).to(device))*(1-x_scaled**2)*get_legendre(n,t_scaled*torch.rand(1).to(device))

def random_legendre2_softplus(X,boundaries):
    n = torch.randint(0,20, (1,)).to(device)
    x = X[:,0]
    t = X[:,1]
    x_scaled = 2*(x-boundaries.x_lb)/(boundaries.x_ub-boundaries.x_lb)-1
    t_scaled = 2*(t-boundaries.t_lb)/(boundaries.t_ub-boundaries.t_lb)-1
    #return torch.nn.functional.softplus(get_legendre(n,x_scaled*torch.rand(1).to(device))*get_legendre(n,t_scaled*torch.rand(1).to(device)))*(x_scaled**4-1)
    return ((get_legendre(n,x_scaled*torch.rand(1).to(device))*get_legendre(n,t_scaled*torch.rand(1).to(device)))**2)*(1-x_scaled**2)

def random_int(min_val, max_val):
        return np.random.randint(min_val, max_val+1)

def random_log_float(min_val, max_val):
        return 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))

def random_float(min_val, max_val):
        return np.random.rand(1)[0]*(max_val-min_val)+min_val


def perturb_log(val, factor = 3.0):
                return 10 ** np.random.uniform(np.log10(val / factor), np.log10(val * factor))

def perturb_linear(val, frac=0.5):
        return random_float(val * (1 - frac), val * (1 + frac))

def perturb_int(val, spread=3):
        return random_int(max(1, val - spread), val + spread)

