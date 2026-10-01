
import torch
import json
import csv

import train_1d
import NN_models.models as models
import NN_models.loss_functions as loss_functions
import datetime
import problems.utils as utils
from pathlib import Path
import problems.euler as euler
from torch import nn
import numpy as np
import copy

class Hyperparams():
        def __init__(self, start_lr = 0.0032770241692461418,
                end_lr = 0.00013657801980498986,#7e-5,#0.0014511622208796802,
                width = 40,
                depth = 7,
                epochs= 500,
                ic_start_loss = 3.9896427578830265,
                ic_end_loss = 3.9896427578830265,
                bc_start_loss = 1.4031313625551958,
                bc_end_loss = 1.4031313625551958,
                pde_start_loss = 5,
                pde_end_loss = 0.01,
                batch_size_ic = 256,
                batch_size_bc = 256,
                batch_size_PDE = 10000,
                max_clip = 4.460749049804084):

                self.start_lr = start_lr
                self.end_lr = end_lr
                self.width = width
                self.depth = depth
                self.epochs = epochs
                self.ic_start_loss = ic_start_loss
                self.ic_end_loss = ic_end_loss
                self.bc_start_loss = bc_start_loss
                self.bc_end_loss = bc_end_loss
                self.pde_start_loss = pde_start_loss
                self.pde_end_loss = pde_end_loss
                self.batch_size_ic = batch_size_ic
                self.batch_size_bc = batch_size_bc
                self.batch_size_PDE = batch_size_PDE
                self.max_clip = max_clip
                self.time= datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        @classmethod
        def from_json(cls, path):
                with open(path, "r") as f:
                        data = json.load(f)
                data.pop("time", None)
                data.pop("final_loss", None)
                data.pop("early_stopped", None)
                return cls(**data)

        def save(self, folder):
                folder.mkdir(parents=True, exist_ok=True)
                with open(folder/"hyperparams.json", "w") as f:
                        json.dump(vars(self), f, indent=4)

#JUST FOR DATA
PRINT_FREQ = 100
TEST_FREQ = 100
STOP_EPOCH = 20000
EARLY_STOP_THRESHOLD = 0.45
SLICES = 10

TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
current_path = Path(__file__).resolve().parent

DATA_PATH = current_path/"data"/"euler_data.mat"

def save_nets(networks, output_folder):
      for i, net in enumerate(networks):
            torch.save(net.state_dict(), output_folder / f"net_{i}.pth")

def run_test(hp = None):
    OUTPUT_PATH_FOLDER = current_path/"output"/"euler_causal_continuous"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    boundaries, test_data,gamma  = euler.read_data(DATA_PATH)
    ic_fn = euler.ic_fn
    networks = []
    for slice in range(SLICES):
        slice_boundaries = copy.copy(boundaries)
        slice_boundaries.t_lb = boundaries.t_lb+(boundaries.t_ub-boundaries.t_lb)*slice/SLICES
        slice_boundaries.t_ub = boundaries.t_lb+(boundaries.t_ub-boundaries.t_lb)*(slice+1)/SLICES    
        slice_start = slice * test_data.t.numel() // SLICES
        slice_stop  = (slice + 1) * test_data.t.numel() // SLICES
        slice_test_data = utils.TestDataset.__new__(utils.TestDataset)
        slice_test_data.x    = test_data.x
        slice_test_data.t    = test_data.t[slice_start:slice_stop]
        slice_test_data.usol = test_data.usol[:, slice_start:slice_stop]
        network = sequence(ic_fn, slice_boundaries, slice_test_data, gamma, OUTPUT_PATH_FOLDER, hp)
        network.eval()
        ic_fn = lambda x, net=network, t=slice_boundaries.t_ub: \
        net(torch.cat([x, torch.full_like(x, t)], dim=1)).detach()
        networks.append(network)
    save_nets(networks, OUTPUT_PATH_FOLDER)
    euler.print_models_output(test_data= test_data, networks=networks, output_path=OUTPUT_PATH_FOLDER, gamma= gamma, slices = SLICES, suffix= "_stich")

def sequence(ic_fn, boundaries, test_data, gamma,output_path, hp = None):
        if hp is None:
            hp = Hyperparams()
        OUTPUT_PATH_FOLDER = output_path/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        hp.save(OUTPUT_PATH_FOLDER)
        print(f"IC LOSS: {hp.ic_start_loss}")

        #MAIN NET
        network = models.FlexibleWeakEulerNet(width = hp.width, depth = hp.depth, activation = nn.Tanh, input = 2, output = 3)
        optimizer = torch.optim.AdamW(network.parameters(), lr=hp.start_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                total_steps=hp.epochs,
                max_lr=hp.start_lr,
                pct_start=0.3,
                anneal_strategy='cos'
                )
        print_fn = lambda suffix = "": euler.print_model_output(test_data= test_data, network=network, output_path=OUTPUT_PATH_FOLDER, gamma = gamma, suffix = suffix)
        pde_fn = lambda X, U:  euler.pde_fn(X, U, gamma)

        #LOSSES MAIN
        IC_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.ic_start_loss, end_loss = hp.ic_end_loss, epochs = hp.epochs)
        BC_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.bc_start_loss, end_loss = hp.bc_end_loss, epochs = hp.epochs)
        PDE_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss=hp.pde_start_loss, end_loss=hp.pde_end_loss, epochs= hp.epochs)
        IC_loss = loss_functions.ICLoss(batch_size = hp.batch_size_ic, boundaries = boundaries, f=ic_fn, network=network, loss_weight_scheduler = IC_loss_weight_scheduler)
        BC_loss = loss_functions.FnBCLoss(batch_size = hp.batch_size_bc, boundaries = boundaries, network = network, bc_fn = euler.bc_fn, loss_weight_scheduler = BC_loss_weight_scheduler)
        PDE_loss = loss_functions.PDELoss(batch_size=hp.batch_size_PDE, boundaries= boundaries, network= network, pde_fn=pde_fn ,loss_weight_scheduler= PDE_loss_weight_scheduler)

        net_trainer = models.NetTrainer(
                losses = [IC_loss , BC_loss, PDE_loss], optimizers = [optimizer],
                schedulers = [scheduler], network = network, max_clip = hp.max_clip,
                print_fn = print_fn)
        euler_validation_loss = lambda pred, usol : euler.euler_validation_loss(pred, usol, gamma)
        loss, early_stopped, last_net = train_1d.train(test_data= test_data, net_trainer = net_trainer, epochs = hp.epochs, print_freq = PRINT_FREQ, test_freq = TEST_FREQ, early_stop_threshold = EARLY_STOP_THRESHOLD, stop_epoch = STOP_EPOCH, test_loss_fn = euler_validation_loss)


        with open(OUTPUT_PATH_FOLDER/"hyperparams.json", "r+") as f:
                data = json.load(f)
                data["final_loss"] = float(loss)
                data["early_stopped"] = early_stopped
                f.seek(0)
                json.dump(data, f, indent=4)

        results_csv = OUTPUT_PATH_FOLDER.parent/"results.csv"
        row = {**vars(hp), "final_loss": float(loss), "early_stopped": early_stopped}
        write_header = not results_csv.exists()
        with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                        writer.writeheader()
                writer.writerow(row)
        return last_net

if __name__ == "__main__":
        run_test()
