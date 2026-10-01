
import torch
import json
import csv

import train_1d
import NN_models.models as models
import NN_models.loss_functions as loss_functions
import datetime
import problems.utils as utils
from pathlib import Path
import problems.advection as advection
from torch import nn
import numpy as np

class Hyperparams():
        def __init__(self, start_lr = 0.006902081649126929,
                end_lr = 0.0006,#0.0014511622208796802
                width = 26,
                depth = 4,
                epochs= 10000,
                ic_start_loss = 4.5478836658150765,
                ic_end_loss = 4.5478836658150765,
                pde_start_loss= 5,
                pde_end_loss = 5,
                bc_start_loss = 1.4031313625551958,
                bc_end_loss = 1.4031313625551958,
                batch_size_ic = 256,
                batch_size_bc = 256,
                batch_size_pde = 9782,
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
                self.batch_size_pde = batch_size_pde
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
STOP_EPOCH = 200
EARLY_STOP_THRESHOLD = 0.35

TIME = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
current_path = Path(__file__).resolve().parent

DATA_PATH = current_path/"data"/"linear_advec_data.mat"


def run_test(hp = None):
        if hp is None:
                hp = Hyperparams()
        OUTPUT_PATH_FOLDER = current_path/"output"/"advection_continuous"/hp.time
        hp.save(OUTPUT_PATH_FOLDER)
        print(f"IC LOSS: {hp.ic_start_loss}")
        boundaries, test_data  = advection.read_data(DATA_PATH)

        #MAIN NET
        network = models.FlexibleNet(width = hp.width, depth = hp.depth, activation = nn.Tanh, input = 2, output = 1)
        optimizer = torch.optim.AdamW(network.parameters(), lr=hp.start_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hp.epochs,
                eta_min=hp.end_lr
                )
        print_fn = lambda suffix = "": models.print_model_output(test_data= test_data, network=network, output_path=OUTPUT_PATH_FOLDER, suffix = suffix)

        #LOSSES MAIN
        PDE_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.pde_start_loss, end_loss = hp.pde_end_loss, epochs = hp.epochs)
        IC_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.ic_start_loss, end_loss = hp.ic_end_loss, epochs = hp.epochs)
        BC_loss_weight_scheduler = loss_functions.LossWeightScheduler(start_loss = hp.bc_start_loss, end_loss = hp.bc_end_loss, epochs = hp.epochs)
        pde_loss = loss_functions.PDELoss(batch_size = hp.batch_size_pde, boundaries = boundaries, pde_fn = advection.pde_fn, network = network, loss_weight_scheduler = PDE_loss_weight_scheduler)
        IC_loss = loss_functions.ICLoss(batch_size = hp.batch_size_ic, boundaries = boundaries, f=advection.ic_fn, network=network, loss_weight_scheduler = IC_loss_weight_scheduler)
        BC_loss = loss_functions.BCFreeOutflowLoss(batch_size = hp.batch_size_bc, boundaries = boundaries, network = network, loss_weight_scheduler = BC_loss_weight_scheduler)

        net_trainer = models.NetTrainer(
                losses = [pde_loss, IC_loss, BC_loss],
                optimizers = [optimizer], schedulers = [scheduler],
                network = network, max_clip = hp.max_clip,
                print_fn = print_fn)

        loss, early_stopped, last_net = train_1d.train(test_data= test_data, net_trainer = net_trainer, epochs = hp.epochs, print_freq = PRINT_FREQ, test_freq = TEST_FREQ, early_stop_threshold = EARLY_STOP_THRESHOLD, stop_epoch = STOP_EPOCH)

        torch.save(last_net.state_dict(), OUTPUT_PATH_FOLDER / "best_net.pth")
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

def random_int(min_val, max_val):
        return np.random.randint(min_val, max_val+1)

def random_log_float(min_val, max_val):
        return 10 ** np.random.uniform(np.log10(min_val), np.log10(max_val))

def random_float(min_val, max_val):
        return np.random.rand(1)[0]*(max_val-min_val)+min_val

if __name__ == "__main__":
        for i in range(1):
                run_test()
