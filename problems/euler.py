import torch
import h5py
from problems import utils
import NN_models.models as models
from torch.utils.data import Dataset
import numpy as np

def pde_fn(X,U, gamma):
    rho = U[:,0]
    v = U[:,1]
    p = U[:,2]

    rho_e = p/(gamma-1) + 1/2 *rho*v**2

    #TODO: consider using jacrev
    #TODO: consider outputting rho, v and e_tot instead of rho_v etc. from U

    d_rho_dX = torch.autograd.grad(rho,X, grad_outputs=torch.ones_like(rho),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    d_rho_v_dX = torch.autograd.grad(rho*v,X, grad_outputs=torch.ones_like(rho),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    d_rho_e_dX = torch.autograd.grad(rho_e,X, grad_outputs=torch.ones_like(rho),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    d_rho_v_sq_p_dX =torch.autograd.grad(rho*v**2 + p,X, grad_outputs=torch.ones_like(rho),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    d_rho_e_p_v_dX = torch.autograd.grad((rho_e + p)*v,X, grad_outputs=torch.ones_like(rho),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)


    d_rho_dt = d_rho_dX[0][:,1]
    d_rho_v_dt = d_rho_v_dX[0][:,1]
    d_rho_e_dt = d_rho_e_dX[0][:,1]


    d_rho_v_dx= d_rho_v_dX[0][:,0]
    d_rho_v_sq_p_dx = d_rho_v_sq_p_dX[0][:,0]
    d_rho_e_p_v_dx = d_rho_e_p_v_dX[0][:,0]


    pde_1 = d_rho_dt + d_rho_v_dx
    pde_2 = d_rho_v_dt+ d_rho_v_sq_p_dx
    pde_3 = d_rho_e_dt + d_rho_e_p_v_dx

    return torch.mean(pde_1**2 + pde_2**2 + pde_3**2)


def weak_integral(X,U, gamma, TF, is_adversarial):

    TF_1 = TF[:,0]
    TF_2 = TF[:,1]
    TF_3 = TF[:,2]


    dTF1_dX = torch.autograd.grad(TF_1,X, grad_outputs=torch.ones_like(TF_1),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    dTF2_dX = torch.autograd.grad(TF_2,X, grad_outputs=torch.ones_like(TF_2),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    dTF3_dX = torch.autograd.grad(TF_3,X, grad_outputs=torch.ones_like(TF_3),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    tf1_x = dTF1_dX[0][:,0]
    tf2_x = dTF2_dX[0][:,0]
    tf3_x = dTF3_dX[0][:,0]

    rho = U[:,0]
    v = U[:,1]
    p = U[:,2]
    rho_e = p/(gamma-1) + 1/2 *rho*v**2

    drho_dX = torch.autograd.grad(rho,X, grad_outputs=torch.ones_like(rho),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    drhov_dX = torch.autograd.grad(rho*v,X, grad_outputs=torch.ones_like(rho),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    drhoe_dX = torch.autograd.grad(rho_e,X, grad_outputs=torch.ones_like(rho_e),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)
    rho_t = drho_dX[0][:,1]
    rhov_t = drhov_dX[0][:,1]
    rhoe_t = drhoe_dX[0][:,1]

    if is_adversarial:
        rho_t  = rho_t.detach()
        rhov_t = rhov_t.detach()
        rhoe_t = rhoe_t.detach()
        rho = rho.detach()
        v = v.detach()
        p = p.detach()
        rho_e  = rho_e.detach()
    else:
        TF_1   = TF_1.detach()
        TF_2   = TF_2.detach()
        TF_3   = TF_3.detach()
        tf1_x  = tf1_x.detach()
        tf2_x  = tf2_x.detach()
        tf3_x  = tf3_x.detach()

    integrand1 = torch.mean(rho_t * TF_1 - rho*v * tf1_x)
    integrand2 = torch.mean(rhov_t * TF_2 - (rho*v**2 + p) * tf2_x)
    integrand3 = torch.mean(rhoe_t * TF_3 - (rho_e + p)*v * tf3_x)


    loss = integrand1+integrand2+integrand3
    if is_adversarial:
        tf1_x_size = (1/2)*torch.mean(tf1_x**2)
        tf2_x_size = (1/2)*torch.mean(tf2_x**2)
        tf3_x_size = (1/2)*torch.mean(tf3_x**2)
        loss = loss - tf1_x_size -tf2_x_size - tf3_x_size

    return loss

def entropy(X,U, gamma, TF, is_adversarial):

    dTF_dX = torch.autograd.grad(TF,X, grad_outputs=torch.ones_like(TF),
                                    retain_graph=True,
                                    create_graph=True,
                                    allow_unused=False)

    tf_x = dTF_dX[0][:,0]

    rho = U[:,0]
    v = U[:,1]
    p = U[:,2]
    m = rho*v

    S = gamma * torch.log(rho) - torch.log(p)

    G = rho*S

    F = m*S

    drhoS_dX = torch.autograd.grad(G,X, grad_outputs=torch.ones_like(G),
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)

    drhovS_dX = torch.autograd.grad(F,X, grad_outputs=torch.ones_like(F),
                                    retain_graph=True,
                                    create_graph=True,
                                    allow_unused=False)

    rhoS_t = drhoS_dX[0][:,1]

    rhovS_x = drhovS_dX[0][:,0]

    if is_adversarial:
        rhoS_t = rhoS_t.detach()
        rhovS_x = rhovS_x.detach()
    else:
        TF = TF.detach()
        tf_x = tf_x.detach()

    #possible sign error?
    loss = torch.mean(torch.relu(rhoS_t + rhovS_x)*TF)
    if is_adversarial:
        tf_x_size = (1/2) * torch.mean(tf_x**2)
        loss = loss - tf_x_size
    return loss



def ic_fn(x):
    x0 = -4
    rho_left = 27 / 7
    v_left = 4 * np.sqrt(35) / 9
    p_left = 31 / 3

    v_right = 0.0
    p_right = 1.0

    rho =   torch.where(x > x0, 1 + 1 / 5 * torch.sin(5 * x), torch.full_like(x,rho_left)) 
    v =  torch.where(x > x0, torch.full_like(x,v_right), torch.full_like(x,v_left))
    p = torch.where(x > x0, p_right, p_left)
    #https://repository.tudelft.nl/record/uuid:6fd86786-153e-4c98-b4e2-8fa36f90eb2a lol
    usol0 = torch.stack([rho,v,p],1)
    return usol0.squeeze(2)

def ic_fn_modified_sod(x):
    # Modified Sod shock tube (Toro Section 6.4), discontinuity at x=0.3
    rho = torch.where(x < 0.3, torch.ones_like(x),       torch.full_like(x, 0.125))
    v   = torch.where(x < 0.3, torch.full_like(x, 0.75), torch.zeros_like(x))
    p   = torch.where(x < 0.3, torch.ones_like(x),       torch.full_like(x, 0.1))
    return torch.stack([rho, v, p], 1).squeeze(2)

def bc_fn_modified_sod(X, network):
    x = X[:,0]
    pred = network(X)
    return torch.mean((ic_fn_modified_sod(x.unsqueeze(1)) - pred)**2)

def bc_fn(X, network):
    x = X[:,0]
    pred = network(X)
    return torch.mean((ic_fn(x.unsqueeze(1))-pred)**2)

class EulerTestDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.x, self.t, self.usol = torch.from_numpy(data['x'].T).to(torch.float64), torch.from_numpy(data['t'].T).to(torch.float64), torch.from_numpy(data['usol']).transpose(0,1).to(torch.float64)
        print(f"usol shape:  {self.usol.shape}")
    def __len__(self):
        return self.usol.shape[0] * self.usol.shape[1]

    def __getitem__(self, idx):
        x_i = idx % self.x.numel()
        t_i = idx //self.x.numel()
        return torch.cat([self.x[x_i],self.t[t_i]]), self.usol[x_i,t_i,:]

def print_model_output(test_data, network, output_path, gamma, suffix = ""):
    x, t, usol = test_data.x,test_data.t,test_data.usol.numpy()

    mesh_x,mesh_t = torch.meshgrid(x.squeeze(),t.squeeze(),indexing="ij")
    X = torch.stack((mesh_x, mesh_t), dim=2)
    X = torch.flatten(X, end_dim=1)
    X = X.to(torch.get_default_device())
    with torch.no_grad():
        u_pred = network(X).cpu().numpy()

    u_pred = u_pred.reshape(x.numel(), t.numel(), 3)
    rho, rho_v, rho_e = convert_model_output(u_pred, gamma)

    models.save_1d_output_graphs(output_path/"rho", usol[:,:,0], rho, mesh_x, mesh_t, suffix)
    models.save_1d_output_graphs(output_path/"rho_v", usol[:,:,1], rho_v, mesh_x, mesh_t, suffix)
    models.save_1d_output_graphs(output_path/"rho_e", usol[:,:,2], rho_e, mesh_x, mesh_t, suffix)

def print_models_output(test_data, networks, output_path, gamma, slices, suffix = ""):
    x, t, usol = test_data.x,test_data.t,test_data.usol.numpy()

    mesh_x, mesh_t = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
    t_chunks = torch.chunk(t.squeeze(), slices)
    u_pred_chunks = []
    for net, t_chunk in zip(networks, t_chunks):
        mesh_x_i, mesh_t_i = torch.meshgrid(x.squeeze(), t_chunk, indexing="ij")
        X_i = torch.stack((mesh_x_i, mesh_t_i), dim=2).flatten(end_dim=1)
        X_i = X_i.to(torch.get_default_device())
        with torch.no_grad():
            u_chunk = net(X_i).cpu().numpy().reshape(x.numel(), t_chunk.numel(),3)
        u_pred_chunks.append(u_chunk)
    u_pred = np.concatenate(u_pred_chunks, axis=1)
    rho, rho_v, rho_e = convert_model_output(u_pred, gamma)

    models.save_1d_output_graphs(output_path/"rho", usol[:,:,0], rho, mesh_x, mesh_t, suffix)
    models.save_1d_output_graphs(output_path/"rho_v", usol[:,:,1], rho_v, mesh_x, mesh_t, suffix)
    models.save_1d_output_graphs(output_path/"rho_e", usol[:,:,2], rho_e, mesh_x, mesh_t, suffix)

def convert_model_output(u_pred, gamma):
    rho = u_pred[:,:,0]
    rho_v = u_pred[:,:,0]*u_pred[:,:,1]
    rho_e = u_pred[:,:,2]/(gamma-1) + 1/2 *rho_v*u_pred[:,:,1]
    return rho, rho_v, rho_e

def euler_validation_loss(pred, usol, gamma):
    rho = pred[:,0]
    rho_v = pred[:,0]*pred[:,1]
    rho_e = pred[:,2]/(gamma-1) + 1/2 *rho_v*pred[:,1]
    converted_pred = torch.stack([rho,rho_v, rho_e],1)
    squared_error = (usol-converted_pred)**2
    mse = torch.mean(squared_error, 0)
    return mse[0]

def read_data(data_path):
    with h5py.File(data_path, 'r') as f:
        b = {key: f[key][()]
         for key in f.keys()}
        gamma = b['gamma_gas'][0]
        boundaries = utils.Boundaries(x_lb = f["x_lb"][0], x_ub = f["x_ub"][0],t_lb = f["t_lb"][0],t_ub = f["t_ub"][0],x_size = int(f["x_size"][0]),t_size = int(f["t_size"][0]))
        test_data = EulerTestDataset(b)

        return boundaries, test_data, gamma
