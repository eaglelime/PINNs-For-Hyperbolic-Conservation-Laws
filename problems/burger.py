
import torch
import h5py
from problems import utils



def entropy(X,U,TF, is_adversarial, c_vec):
    TF = torch.nn.functional.softplus(TF, beta=1.0)

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
        integrand_max = torch.tensor(0.0, requires_grad=True)

    U = U.flatten().unsqueeze(0) 
    c = c_vec.unsqueeze(1)
    mu = 0.01
    u_minus_c = U - c
    sign_approx = lambda x: 2 * torch.sigmoid(x / mu) - 1
    abs_approx = lambda x: 2 * mu * torch.log(0.5 * (1 + torch.exp(x / mu))) - x

    Q = sign_approx(u_minus_c) * (U**2 - c**2) / 2.0

    integrand = abs_approx(u_minus_c) * tf_t.unsqueeze(0) + Q * tf_x.unsqueeze(0)
    norm = torch.mean(tf_x**2 + tf_t**2)+1e-8
    if is_adversarial:
        loss = torch.max(-torch.mean(integrand, dim=1))
        norm = torch.sqrt(norm)
    else:
        loss = torch.max(torch.relu(-torch.mean(integrand, dim=1) ) ** 2)
    return loss/ norm + 1e-8


def new_entropy(X,U,TF, is_adversarial):
    U = U.squeeze(1)
    dTF_dX = torch.autograd.grad(TF,X, grad_outputs=torch.ones_like(TF),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    tf_x = dTF_dX[0][:,0]

    eta = (1/2)*U**2
    q = (1/3)*U**3

    dq_dX = torch.autograd.grad(eta,X, grad_outputs=torch.ones_like(eta),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    deta_dX = torch.autograd.grad(eta,X, grad_outputs=torch.ones_like(eta),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    eta_t = deta_dX[0][:,1]
    q_x = dq_dX[0][:,0]


    if is_adversarial:
        eta_t = eta_t.detach()
        q_x = q_x.detach()
    else:
        tf_x = tf_x.detach()
        TF = TF.detach()

    #maybe flip q
    integrand = torch.relu(eta_t + q_x)*TF
    loss = torch.mean(integrand)
    if is_adversarial:
        tf_x_size = (1/2)*torch.mean(tf_x**2)
        loss = loss-tf_x_size
    return loss


def weak_integral(X,U, TF, is_adversarial):
    U = U.squeeze(1)
    dTF_dX = torch.autograd.grad(TF,X, grad_outputs=torch.ones_like(TF),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    tf_x = dTF_dX[0][:,0]

    dU_dX = torch.autograd.grad(U,X, grad_outputs=torch.ones_like(U),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    u_t = dU_dX[0][:,1]

    if is_adversarial:
        U = U.detach()
        u_t = u_t.detach()
    else:
        tf_x = tf_x.detach()
        TF = TF.detach()

    integrand = torch.mean(u_t * TF - ((U**2)/2) * tf_x)

    if is_adversarial:
        tf_x_size = (1/2)*torch.mean(tf_x**2)
        return integrand-tf_x_size
    return integrand

def pde_fn(X,U):
    dU_dX = torch.autograd.grad(U,X, grad_outputs=torch.ones_like(U),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    u_x = dU_dX[0][:,0]
    dU_dxX = torch.autograd.grad(u_x,X, grad_outputs=torch.ones_like(u_x),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    u_t = dU_dX[0][:,1]
    u_xx = dU_dxX[0][:,0]
    pde = u_t+U.squeeze()*u_x #+ 0.001*u_xx
    return torch.mean(pde**2)

def ic_fn(x):
    return 2+ torch.sin(torch.pi*(2*x)).detach()

def ic_shock_fn(x):
    return torch.where(x < 0, torch.tensor(1.0), torch.tensor(0.0))


def read_data(data_path):
    with h5py.File(data_path, 'r') as f:
        b = {key: f[key][()]
         for key in f.keys()}
        boundaries = utils.Boundaries(x_lb = f["x_lb"][0], x_ub = f["x_ub"][0],t_lb = f["t_lb"][0],t_ub = f["t_ub"][0],x_size = int(f["x_size"][0]),t_size = int(f["t_size"][0]))
        test_data = utils.TestDataset(b)
    return boundaries, test_data
