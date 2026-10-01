import scipy
import torch
import h5py
from problems import utils
from problems.generate_linear_advection_data import C

def entropy(X,U,TF, is_adversarial, c_vec):


    TF = torch.nn.functional.softplus(TF, beta=1.0)
    TF = TF.unsqueeze(1)
    dTF_dX = torch.autograd.grad(TF,X, grad_outputs=torch.ones_like(TF),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    tf_x = dTF_dX[0][:,0]
    tf_t = dTF_dX[0][:,1]

    if is_adversarial:
        U = U.detach()
    else:
        tf_x = tf_x.detach()
        tf_t = tf_t.detach()

    U = U.flatten().unsqueeze(0) 
    c = c_vec.unsqueeze(1)
    mu = 0.01
    u_minus_c = U - c
    sign_approx = lambda x: 2 * torch.sigmoid(x / mu) - 1
    abs_approx = lambda x: 2 * mu * torch.log(0.5 * (1 + torch.exp(x / mu))) - x

    Q = sign_approx(u_minus_c) * C*(U - c)

    integrand = abs_approx(u_minus_c) * tf_t.unsqueeze(0) + Q * tf_x.unsqueeze(0)
    norm = torch.mean(tf_x**2 + tf_t**2)+1e-8
    if is_adversarial:
        loss = torch.max(-torch.mean(integrand, dim=1))
        norm = torch.sqrt(norm)
    else:
        loss = torch.max(torch.relu(-torch.mean(integrand, dim=1) ) ** 2)
    return loss/ norm + 1e-8

def pde_fn(X,U):
    dU_dX = torch.autograd.grad(U,X, grad_outputs=torch.ones_like(U),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    u_x = dU_dX[0][:,0]
    u_t = dU_dX[0][:,1]

    pde = u_t+C*u_x
    return torch.mean(pde**2)

def ic_fn(x):
    #return torch.sin(torch.pi*x)
    return torch.where(torch.sin(torch.pi*x)>0,1,0).detach()

def ic_shock_fn(x):
    return torch.where(x < 0, torch.tensor(1.0), torch.tensor(0.0))


def read_data(data_path):
    data = scipy.io.loadmat(data_path,appendmat=False)
    b = data["boundaries"]
    boundaries = utils.Boundaries(x_lb = float(b["x_lb"].item()[0][0]), x_ub = float(b["x_ub"].item()[0][0]),t_lb = float(b["t_lb"].item()[0][0]),t_ub = float(b["t_ub"].item()[0][0]),x_size = int(b["x_size"].item()[0][0]),t_size = int(b["t_size"].item()[0][0]))
    test_data = utils.TestDataset(data)
    return boundaries, test_data
