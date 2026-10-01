from torch import nn
import torch
import numpy as np
from scipy.stats import qmc
from torch.distributions import Cauchy

"""l1_loss = self.l1_strength* sum(p.abs().sum() for p in params) 
        l2_loss = self.l2_strength* sum((p**2).sum() for p in params) 
        density_f = lambda p: (1/(torch.pi*Y)) * (Y**2/(p**2 + Y**2))
        params = [p for name, p in params if 'weight' in name]
        """

class CauchDensityLoss():
    def __init__(self, network, loss_weight_scheduler, gamma):
        self.network = network
        self.loss_weight_scheduler = loss_weight_scheduler
        self.gamma = gamma
        self.distr = Cauchy(loc=0, scale=gamma)
    def get_loss(self, epoch, print_loss):
        named_params = self.network.named_parameters()
        params = [p for name, p in named_params if 'weight' in name]
        cauchy_density_loss = - sum(self.distr.log_prob(p).sum() for p in params)/sum(p.numel() for p in params)
        total_loss = cauchy_density_loss * self.loss_weight_scheduler.get_weight(epoch)
        if print_loss:
            print(f"Cauchy density loss: {total_loss}")
        return total_loss



class BCLoss():
    def __init__(self, batch_size, boundaries, network, loss_weight_scheduler):
        self.sampler = qmc.LatinHypercube(d=1)
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.network = network
        self.loss_weight_scheduler = loss_weight_scheduler

    def get_loss(self, epoch, print_loss):
        sample = self.sampler.random(n=self.batch_size)
        sample = torch.from_numpy(qmc.scale(sample, self.boundaries.t_lb, self.boundaries.t_ub)).to(torch.float64).to(torch.get_default_device())
        X_lb = torch.stack([torch.full((self.batch_size,1), self.boundaries.x_lb), sample], 1).squeeze(2)
        X_ub = torch.stack([torch.full((self.batch_size,1), self.boundaries.x_ub), sample], 1).squeeze(2).detach()
        X_lb.requires_grad_()
        X_ub.requires_grad_()
        pred_ub = self.network(X_ub)
        pred_lb = self.network(X_lb)
        weight = self.loss_weight_scheduler.get_weight(epoch)
        final_loss = weight*torch.mean((pred_ub-pred_lb)**2)
        if print_loss:
            print(f"BC loss: {final_loss}")
        return final_loss


class BCFreeOutflowLoss():
    def __init__(self, batch_size, boundaries, network, loss_weight_scheduler):
        self.sampler = qmc.LatinHypercube(d=1)
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.network = network
        self.loss_weight_scheduler = loss_weight_scheduler

    def get_loss(self, epoch, print_loss):
        sample = self.sampler.random(n=self.batch_size)
        sample = torch.from_numpy(qmc.scale(sample, self.boundaries.t_lb, self.boundaries.t_ub)).to(torch.float64).to(torch.get_default_device())
        X_lb = torch.stack([torch.full((self.batch_size,1), self.boundaries.x_lb), sample], 1).squeeze(2)
        X_ub = torch.stack([torch.full((self.batch_size,1), self.boundaries.x_ub), sample], 1).squeeze(2).detach()
        X_lb.requires_grad_()
        #X_ub.requires_grad_()
        with torch.no_grad():
            pred_ub = self.network(X_ub)
        pred_lb = self.network(X_lb)
        weight = self.loss_weight_scheduler.get_weight(epoch)
        final_loss = weight*torch.mean((pred_ub-pred_lb)**2)
        if print_loss:
            print(f"BC loss: {final_loss}")
        return final_loss


class FnBCLoss():
    def __init__(self, batch_size, boundaries, network, bc_fn, loss_weight_scheduler):
        self.sampler = qmc.LatinHypercube(d=1)
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.bc_f = bc_fn
        self.network = network
        self.loss_weight_scheduler = loss_weight_scheduler

    def get_loss(self, epoch, print_loss):
        batch_size_per_side = self.batch_size//2
        sample_lb = self.sampler.random(n=batch_size_per_side)
        sample_ub = self.sampler.random(n=batch_size_per_side)
        sample_lb = torch.from_numpy(qmc.scale(sample_lb, self.boundaries.t_lb, self.boundaries.t_ub)).to(torch.get_default_device()).to(torch.float64)
        sample_ub = torch.from_numpy(qmc.scale(sample_ub, self.boundaries.t_lb, self.boundaries.t_ub))
        sample_ub = sample_ub.to(torch.float64).to(torch.get_default_device())
        X_lb = torch.stack([torch.full((batch_size_per_side,1), self.boundaries.x_lb), sample_lb], 1).squeeze(2)
        X_ub = torch.stack([torch.full((batch_size_per_side,1), self.boundaries.x_ub), sample_ub], 1).squeeze(2)
        X = torch.cat([X_lb,X_ub],0)
        X.requires_grad_()
        weight = self.loss_weight_scheduler.get_weight(epoch)
        final_loss = weight * self.bc_f(X,self.network)
        if print_loss:
            print(f"BC loss: {final_loss}")
        return final_loss

class ICLoss():
    def __init__(self, batch_size, boundaries, f, network, loss_weight_scheduler):
        self.sampler = qmc.LatinHypercube(d=1)
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.f = f
        self.network = network
        self.loss_weight_scheduler = loss_weight_scheduler
    def get_loss(self, epoch, print_loss):
        sample = self.sampler.random(n=self.batch_size)
        x = torch.from_numpy(qmc.scale(sample, self.boundaries.x_lb, self.boundaries.x_ub)).to(torch.float64).to(torch.get_default_device())
        usol = self.f(x)#.detach()
        t0 = self.boundaries.t_lb
        X = torch.stack([x, torch.full((self.batch_size, 1), t0)], 1).squeeze(2)
        net_ic = self.network(X)
        weight = self.loss_weight_scheduler.get_weight(epoch)
        final_loss = weight * torch.mean((net_ic - usol)**2)
        if print_loss:
            print(f"IC loss: {final_loss}")
        return final_loss

class PDELoss():
    def __init__(self, batch_size, boundaries, pde_fn, network, loss_weight_scheduler):
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.pde_fn = pde_fn
        self.sampler = qmc.LatinHypercube(d=2)
        self.network = network
        self.loss_weight_scheduler = loss_weight_scheduler

    def get_loss(self, epoch, print_loss):
        #sample = self.sampler.random(n=self.batch_size)
        #sample = qmc.scale(sample,[self.boundaries.x_lb,self.boundaries.t_lb],[self.boundaries.x_ub,self.boundaries.t_ub])
        #X = torch.from_numpy(sample).to(torch.float64)
        sample = torch.rand(self.batch_size,2)
        sample = sample*torch.tensor([self.boundaries.x_ub-self.boundaries.x_lb, self.boundaries.t_ub-self.boundaries.t_lb])
        sample = sample + torch.tensor([self.boundaries.x_lb,self.boundaries.t_lb])
        X = sample
        X.requires_grad_()
        pred= self.network(X)
        pde_loss = self.pde_fn(X,pred)
        weight = self.loss_weight_scheduler.get_weight(epoch)
        final_loss = weight * pde_loss
        if print_loss:
            print(f"PDE loss: {final_loss}")
        return final_loss

class EntropyLoss():
    def __init__(self, batch_size, boundaries, entropy, network, test_function, is_adversarial, loss_weight_scheduler, loss_name = "entropy"):
        self.batch_size = batch_size
        self.boundaries = boundaries
        self.entropy = entropy
        self.sampler = qmc.LatinHypercube(d=2)
        self.network = network
        self.test_function = test_function
        self.is_adversarial = is_adversarial
        self.loss_weight_scheduler = loss_weight_scheduler
        self.loss_name = loss_name

    def get_loss(self, epoch, print_loss):
        #sample = self.sampler.random(n=self.batch_size)
        #sample = qmc.scale(sample,[self.boundaries.x_lb,self.boundaries.t_lb],[self.boundaries.x_ub,self.boundaries.t_ub])
        sample = torch.rand(self.batch_size,2)
        sample = sample*torch.tensor([self.boundaries.x_ub-self.boundaries.x_lb, self.boundaries.t_ub-self.boundaries.t_lb])
        sample = sample + torch.tensor([self.boundaries.x_lb,self.boundaries.t_lb])
        X = sample
        X.requires_grad_()
        pred = self.network(X)
        pred_adv = self.test_function(X)
        entropy_loss = self.entropy(X,pred,pred_adv, self.is_adversarial)
        weight = self.loss_weight_scheduler.get_weight(epoch)
        final_loss = weight*entropy_loss
        if print_loss:
            print(f"{self.loss_name} loss: {final_loss}")
        return final_loss

class TestDataLoss():
    def __init__(self, batch_size, test_data, network, loss_weight_scheduler):
        if batch_size == 0:
            return
        self.network = network
        sample_points = qmc.LatinHypercube(d=2).random(n=batch_size)
        indices = torch.from_numpy(qmc.scale(sample_points,[0,0], [test_data.x.numel(),test_data.t.numel()])).to(torch.int32)
        self.loss_weight_scheduler = loss_weight_scheduler
        self.usol = test_data.usol[indices[:,0],indices[:,1]].to(torch.get_default_device())
        self.X = torch.cat([test_data.x[indices[:,0]], test_data.t[indices[:,1]]],1).to(torch.get_default_device())


    def get_loss(self, epoch, print_loss):
        pred = self.network(self.X).squeeze()
        loss = torch.mean((pred-self.usol)**2)
        total_loss = self.loss_weight_scheduler.get_weight(epoch) * loss
        if print_loss:
            print(f"Test data loss: {total_loss}")
        return total_loss

class BCDataLoss():
    def __init__(self, boundaries, test_data, network, loss_weight_scheduler):
        self.network = network
        self.usol = test_data.usol[0,:].unsqueeze(1)
        t = test_data.t
        self.X_ub = torch.stack([torch.full_like(t, boundaries.x_ub), t], dim=1).to(torch.get_default_device())
        self.X_lb = torch.stack([torch.full_like(t, boundaries.x_lb), t], dim=1).to(torch.get_default_device())
        self.loss_weight_scheduler = loss_weight_scheduler

    def get_loss(self, epoch, print_loss):
        pred_ub = self.network(self.X_ub)
        pred_lb = self.network(self.X_lb)
        usol_on_device = self.usol.to(torch.get_default_device())
        loss = torch.mean((pred_ub-usol_on_device)**2) + torch.mean((pred_lb-usol_on_device)**2)
        total_loss = self.loss_weight_scheduler.get_weight(epoch) * loss
        if print_loss:
            print(f"BC data loss: {total_loss}")
        return total_loss

class TransferLoss():
    def __init__(self, batch_size, boundaries, network, transfer_net, loss_weight_scheduler):
        if batch_size == 0:
            return
        self.batch_size = batch_size
        self.network = network
        self.loss_weight_scheduler = loss_weight_scheduler
        self.boundaries = boundaries
        self.transfer_net = transfer_net

    def get_loss(self, epoch, print_loss):
        sample = torch.rand(self.batch_size,2)
        sample = sample*torch.tensor([self.boundaries.x_ub-self.boundaries.x_lb, self.boundaries.t_ub-self.boundaries.t_lb])
        sample = sample + torch.tensor([self.boundaries.x_lb,self.boundaries.t_lb])
        X = sample
        X.requires_grad_()
        pred = self.network(X).squeeze()
        with torch.no_grad():
            transfer_pred = self.transfer_net(X).squeeze()
        loss = torch.mean((pred-transfer_pred)**2)
        total_loss = self.loss_weight_scheduler.get_weight(epoch) * loss
        if print_loss:
            print(f"Transfer loss: {total_loss}")
        return total_loss

class TransferBCLoss():
    def __init__(self, batch_size, boundaries, network, transfer_net, loss_weight_scheduler):
        if batch_size == 0:
            return
        self.batch_size = batch_size
        self.network = network
        self.loss_weight_scheduler = loss_weight_scheduler
        self.boundaries = boundaries
        self.transfer_net = transfer_net
        self.sampler = qmc.LatinHypercube(d=1)

    def get_loss(self, epoch, print_loss):
        sample = self.sampler.random(n=self.batch_size)
        sample = torch.from_numpy(qmc.scale(sample, self.boundaries.t_lb, self.boundaries.t_ub)).to(torch.float64).to(torch.get_default_device())
        X_lb = torch.stack([torch.full((self.batch_size,1), self.boundaries.x_lb), sample], 1).squeeze(2)
        X_ub = torch.stack([torch.full((self.batch_size,1), self.boundaries.x_ub), sample], 1).squeeze(2)
        X_lb.requires_grad_()
        X_ub.requires_grad_()
        pred_ub = self.network(X_ub)
        pred_lb = self.network(X_lb)
        with torch.no_grad():
            transfer_ub = self.transfer_net(X_ub)
            transfer_lb = self.transfer_net(X_lb)
        weight = self.loss_weight_scheduler.get_weight(epoch)
        ub_loss = weight*torch.mean((pred_ub-transfer_ub)**2)
        lb_loss = weight*torch.mean((pred_lb - transfer_lb)**2)
        tot_loss = ub_loss + lb_loss
        if print_loss:
            print(f"Transfer BC loss: {tot_loss}")
        return tot_loss


class LossWeightScheduler():
    def __init__(self, start_loss, end_loss, epochs):
        self.start_loss = start_loss
        self.end_loss = end_loss
        self.epochs = epochs

    def get_weight(self, epoch):
        progress = epoch/self.epochs
        return np.cos(progress*np.pi/2)*(self.start_loss-self.end_loss)+self.end_loss



def loss_weight_scheduler(IC_loss, PDE_loss, BC_loss, mesh_loss, epochs, epoch):
    start_IC, end_IC  = 5,5
    start_PDE, end_PDE = 1,1
    start_BC, end_BC = 1,1
    start_mesh, end_mesh = 5,5

