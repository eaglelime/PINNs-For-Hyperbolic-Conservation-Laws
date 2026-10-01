
import torch
import json
import csv

import train_1d
import NN_models.models as models
import NN_models.loss_functions as loss_functions
import datetime
import problems.utils as utils
from pathlib import Path
import problems.burger as burger
from torch import nn
import numpy as np
import copy

class Hyperparams():

        def __init__(self, max_lr = 0.005857279603702579,
                pct_start = 0.3,
                adv_start_lr = 0.004032516388731619,
                adv_end_lr = 0.0011952131360040987,
                width = 15,
                depth = 6,
                adv_width = 13,
                adv_depth = 3,
                epochs= 500,
                ic_start_loss = 3.089664954237589,
                ic_end_loss = 3.089664954237589,
                entropy_start_loss= 8,#490.23916642242733,
                entropy_end_loss = 8,#490.23916642242733,
                adv_entropy_start_loss = 33.41801898881988,
                adv_entropy_end_loss = 33.41801898881988,
                bc_start_loss = 0.9306470934048028,
                bc_end_loss = 0.9306470934048028,
                batch_size_ic = 256,
                batch_size_bc = 256,
                batch_size_entropy = 9117,
                adv_training_per_epoch = 10,#4,#13,
                max_clip = 6.2398863637916495,
                adv_max_clip = 0.23197233169193587,
                c_samples = 512,
                adv_reset_freq = 328,
                adv_networks = 1
                ):

                self.max_lr= max_lr
                self.pct_start = pct_start
                self.adv_start_lr = adv_start_lr
                self.adv_end_lr = adv_end_lr
                self.width = width
                self.depth = depth
                self.adv_width = adv_width
                self.adv_depth = adv_depth
                self.epochs = epochs
                self.ic_start_loss = ic_start_loss
                self.ic_end_loss = ic_end_loss
                self.entropy_start_loss = entropy_start_loss
                self.entropy_end_loss = entropy_end_loss
                self.adv_entropy_start_loss = adv_entropy_start_loss
                self.adv_entropy_end_loss = adv_entropy_end_loss
                self.bc_start_loss = bc_start_loss
                self.bc_end_loss = bc_end_loss
                self.batch_size_ic = batch_size_ic
                self.batch_size_bc = batch_size_bc
                self.batch_size_entropy = batch_size_entropy
                self.adv_training_per_epoch = adv_training_per_epoch
                self.max_clip = max_clip
                self.adv_max_clip = adv_max_clip
                self.c_samples = c_samples
                self.adv_reset_freq = adv_reset_freq
                self.adv_networks = adv_networks
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
STOP_EPOCH = 200
EARLY_STOP_THRESHOLD = 0.85
SLICES = 3

TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
current_path = Path(__file__).resolve().parent

DATA_PATH = current_path/"data"/"burger_data.mat"

def run_test(hp = None):
    OUTPUT_PATH_FOLDER = current_path/"output"/"weak_burger_continuous_periodic"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    boundaries, test_data  = burger.read_data(DATA_PATH)
    ic_fn = burger.ic_fn
    transfer_net = models.FlexibleNet(width = 15, depth = 6, activation = nn.Tanh, input = 2, output = 1)
    transfer_net.load_state_dict(torch.load(current_path/"saved_nets/best_net.pth"))
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
        network = sequence(ic_fn, slice_boundaries, slice_test_data, OUTPUT_PATH_FOLDER, transfer_net, None)
        network.eval()
        ic_fn = lambda x, net=network, t=slice_boundaries.t_ub: \
        net(torch.cat([x, torch.full_like(x, t)], dim=1)).detach()
        networks.append(network)
    models.print_models_output(test_data= test_data, networks=networks, output_path=OUTPUT_PATH_FOLDER, suffix= "_stich")


def sequence(ic_fn, boundaries, test_data,output_path, transfer_net, hp = None):
        if hp is None:
                hp = Hyperparams()
        OUTPUT_PATH_FOLDER = output_path/datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        hp.save(OUTPUT_PATH_FOLDER)
        print(f"IC LOSS: {hp.ic_start_loss}")
        c_min = torch.min(burger.ic_fn(torch.linspace(boundaries.x_lb,boundaries.x_ub,1000)))
        c_max = torch.max(burger.ic_fn(torch.linspace(boundaries.x_lb,boundaries.x_ub,1000)))
        c_vec = torch.linspace(c_min,c_max,hp.c_samples)
        entropy_fn = lambda X, U, U_adv, detach_derivatives: burger.entropy(X,U,U_adv,detach_derivatives, c_vec)

        #ADV NET

        def init_adv_training(neu_n):
                optim= torch.optim.AdamW(neu_n.parameters(), lr=hp.adv_start_lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optim,
                        T_max=hp.adv_reset_freq*hp.adv_training_per_epoch,
                        eta_min=hp.adv_end_lr
                        )
                neu_n.apply(neu_n._init_weights)
                neu_n.train()
                return [optim], [sched]

        #MAIN NET
        network = models.FlexibleNetPeriodicX(width = hp.width, depth = hp.depth, activation = nn.ReLU, input = 3, output = 1, x_size=np.abs(boundaries.x_ub-boundaries.x_lb))
        optimizer = torch.optim.AdamW(network.parameters(), lr=hp.max_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                total_steps=hp.epochs,
                max_lr=hp.max_lr,
                pct_start= hp.pct_start,
                anneal_strategy='cos'
                )
        print_fn = lambda suffix = "": models.print_model_output(test_data= test_data, network=network, output_path=OUTPUT_PATH_FOLDER, suffix = suffix)

        #LOSSES MAIN
        entropy_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.entropy_start_loss/hp.adv_networks, end_loss = hp.entropy_end_loss/hp.adv_networks, epochs = hp.epochs)
        IC_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.ic_start_loss, end_loss = hp.ic_end_loss, epochs = hp.epochs)
        BC_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.bc_start_loss, end_loss = hp.bc_end_loss, epochs = hp.epochs)
        IC_loss = loss_functions.ICLoss(batch_size = hp.batch_size_ic, boundaries = boundaries, f=ic_fn, network=network, loss_weight_scheduler = IC_loss_weight_scheduler)
        BC_loss = loss_functions.BCLoss(batch_size = hp.batch_size_bc, boundaries = boundaries, network = network, loss_weight_scheduler = BC_loss_weight_scheduler)

        entropy_losses = []
        adv_trainers = []

        for i in range(hp.adv_networks):
                adv_network = models.FlexibleNet(width = hp.adv_width, depth = hp.adv_depth, activation = nn.Tanh, input = 2, output = 1)
                def adv_func(X, net=adv_network):
                        xl, xu = boundaries.x_lb, boundaries.x_ub
                        tl, tu = boundaries.t_lb, boundaries.t_ub
                        x_scale = (2*(X[:,0]-xl)/(xu-xl)-1)**2-1
                        t_scale = (2*(X[:,1]-tl)/(tu-tl)-1)**2-1
                        return net(X).squeeze()* x_scale * t_scale
                def adv_print_fn(f=adv_func, idx=i):
                        models.print_test_network_output(
                                test_function=f, save_path=OUTPUT_PATH_FOLDER,
                                boundaries=boundaries, suffix=idx)
                entropy_loss = loss_functions.EntropyLoss(batch_size = hp.batch_size_entropy, boundaries = boundaries, entropy = entropy_fn, network = network, test_function = adv_func, is_adversarial = False, loss_weight_scheduler = entropy_loss_weight_scheduler)
                adv_entropy_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.adv_entropy_start_loss, end_loss = hp.adv_entropy_end_loss, epochs = hp.epochs)
                adv_entropy_loss = loss_functions.EntropyLoss(batch_size = hp.batch_size_entropy, boundaries = boundaries, entropy = entropy_fn, network = network, test_function = adv_func, is_adversarial = True, loss_weight_scheduler = adv_entropy_loss_weight_scheduler)
                adv_net_trainer = models.AdvNetTrainer(
                        training_per_epoch = hp.adv_training_per_epoch, entropy_loss = adv_entropy_loss, losses = [], init_training=init_adv_training,
                        network = adv_network, max_clip = hp.adv_max_clip,
                        print_fn = adv_print_fn, reset_frequency=hp.adv_reset_freq, offset = i*hp.adv_reset_freq//hp.adv_networks)
                entropy_losses.append(entropy_loss)
                adv_trainers.append(adv_net_trainer)

        net_trainer = models.NetTrainer(
                losses = [IC_loss]+ entropy_losses, optimizers = [optimizer],
                schedulers = [scheduler], network = network, max_clip = hp.max_clip,
                print_fn = print_fn)

        losses, training_losses, early_stopped, best_net, best_net_loss = train_1d.min_max_train(test_data= test_data, net_trainer = net_trainer, adv_trainers = adv_trainers, epochs = hp.epochs, print_freq = PRINT_FREQ, test_freq = TEST_FREQ, early_stop_threshold = EARLY_STOP_THRESHOLD, stop_epoch = STOP_EPOCH)

        torch.save(best_net.state_dict(), OUTPUT_PATH_FOLDER / "best_net.pth")
        utils.plot_losses(losses, OUTPUT_PATH_FOLDER / "losses.png")
        utils.plot_losses(training_losses,OUTPUT_PATH_FOLDER/"training_losses.png")
        with open(OUTPUT_PATH_FOLDER/"hyperparams.json", "r+") as f:
                data = json.load(f)
                data["final_loss"] = float(best_net_loss)
                data["early_stopped"] = early_stopped
                f.seek(0)
                json.dump(data, f, indent=4)

        results_csv = OUTPUT_PATH_FOLDER.parent/"results.csv"
        row = {**vars(hp), "final_loss": float(best_net_loss), "early_stopped": early_stopped}
        write_header = not results_csv.exists()
        with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                        writer.writeheader()
                writer.writerow(row)
        return best_net

if __name__ == "__main__":
        run_test()
