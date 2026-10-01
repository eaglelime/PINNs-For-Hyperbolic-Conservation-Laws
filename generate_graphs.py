import NN_models.models as models

import torch
from torch import nn
from pathlib import Path

import problems.advection as advection
import problems.burger as burger
import problems.euler as euler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from torch.utils.data import DataLoader

current_path = Path(__file__).resolve().parent

MODEL_NAME = "wPINN"

save_path = current_path/"graphs_for_report"/"i_just_want_burger_graph"

DATA_PATH = current_path/"data"/"burger_data.mat"

#NET_PATH = current_path/"saved_nets/burger_PINN.pth" # ex
NET_PATH = current_path/"saved_nets/burger_PINN_causal"

T_TARGETS = []

FILENAME = "euler_wpinn_causal_2"

data_reader = burger.read_data

CAUSAL = True
SLICES = 10

EULER = False

def generate_graph(pred, x, t, usol, save_path, filename):
    save_path.mkdir(parents=True, exist_ok=True)

    x_np = x.squeeze().cpu().numpy() if isinstance(x, torch.Tensor) else x.squeeze()
    t_np = t.squeeze().cpu().numpy() if isinstance(t, torch.Tensor) else t.squeeze()

    for target in T_TARGETS:
        t_str = str(target).replace('.', '')
        complete_filename = f"{filename}{t_str}.svg"

        t_index = np.argmin(np.abs(t_np - target))

        plt.figure(figsize=(10, 6))
        plt.plot(x_np, pred[:, t_index], label=f'{MODEL_NAME} solution')
        plt.plot(x_np, usol[:, t_index], label='Reference solution', color='red')

        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path / complete_filename, format='svg')
        plt.close()

def save_1d_output_graphs(save_path, usol, u_pred, mesh_x, mesh_t, suffix = ""):
    save_path.mkdir(parents=True, exist_ok=True)

    u_diff = np.subtract(usol,u_pred)

    v_min = np.min([np.min(usol),np.min(u_pred),np.min(u_diff)])
    v_max = np.max([np.max(usol),np.max(u_pred),np.max(u_diff)])
    # 2. Create a small separate figure for the standalone colorbar
    fig, ax = plt.subplots(figsize=(8.0, 0.3)) # Adjust aspect ratio (width, height)

    # 3. Set up Normalization and ScalarMappable
    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    sm = cm.ScalarMappable(cmap='rainbow', norm=norm)
    sm.set_array([])

    # 4. Draw the colorbar in the standalone figure
    cbar = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cbar.set_label(r'$u(x, t)$', fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    # 5. Tight layout and save as a separate image
    plt.tight_layout()
    plt.savefig(save_path/"colorbar.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    def plot_and_save(data, title, filename, colorbar = False, axline = True):
        plt.figure(figsize=(10, 6))
        cp = plt.pcolormesh(mesh_t, mesh_x, data, cmap='rainbow', 
                            shading='nearest', vmin=v_min, vmax=v_max)
        cp.set_rasterized(True)
        if axline:
            for target in T_TARGETS:
                plt.axvline(x=target, color="black", linestyle="--", linewidth=1, alpha=0.8)
        if colorbar:
            plt.colorbar(cp).set_label('$u(x, t)$', rotation=0, labelpad=15)
        plt.xlabel('t')
        plt.ylabel('x')
        #plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path / filename, format='pdf', dpi=300)
        plt.close()

    plot_and_save(u_diff, f'{MODEL_NAME} Model Error Residue', "output_diff.pdf", axline = False)
    plot_and_save(u_pred, f'{MODEL_NAME} Model Prediction', f"output{suffix}.pdf")
    plot_and_save(usol, 'Test Data', "test_data.pdf")


def calculate_relative_error(pred, usol):
    u_diff = np.subtract(usol,pred)
    print(np.isnan(usol).any())
    print(np.isinf(usol).any())
    u_diff_norm = np.linalg.norm(u_diff,2)
    usol_norm = np.linalg.norm(usol,2)
    print(f"u diff norm: {u_diff_norm}")
    print(f"usol norm: {usol_norm}")
    relative_error = u_diff_norm/usol_norm
    print(f"relative error: {relative_error}")

if __name__ == "__main__":
    plt.rcParams.update({
        'font.size': 20,          # General text size
        'axes.labelsize': 20,     # Axis label size (e.g., X and Y axis titles)
        'xtick.labelsize': 20,    # Numbers/ticks on the X-axis
        'ytick.labelsize': 20     # Numbers/ticks on the Y-axis
    })
    T = [0.1]
    boundaries, test_data, gamma = None, None,None
    if EULER:
        boundaries, test_data, gamma = data_reader(DATA_PATH)
    else:
        boundaries, test_data  = data_reader(DATA_PATH)
    #x_ub = boundaries.x_ub
    #x_lb = boundaries.x_lb
    #x = np.linspace(x_lb, x_ub, 1000)
    test_data_loader = DataLoader(test_data,batch_size = len(test_data))

    x = 0
    t = 0
    X = 0
    usol = 0
    pred = 0
    mesh_x = 0
    mesh_t = 0
    if EULER:
        if CAUSAL:
            x, t, usol = test_data.x,test_data.t,test_data.usol.cpu().numpy()
            networks = []
            for i in range(SLICES):
                loaded_net = models.FlexibleNet(width = 40, depth = 7, activation = nn.Tanh, input = 2, output = 3)
                loaded_net.load_state_dict(torch.load(NET_PATH/f"net_{i}.pth"))
                networks.append(loaded_net)

            mesh_x, mesh_t = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
            X = torch.stack((mesh_x, mesh_t), dim=2)
            X = torch.flatten(X, end_dim=1)
            X = X.to(torch.get_default_device())
            t_chunks = torch.chunk(t.squeeze(), SLICES)
            u_pred_chunks = []
            for net, t_chunk in zip(networks, t_chunks):
                mesh_x_i, mesh_t_i = torch.meshgrid(x.squeeze(), t_chunk, indexing="ij")
                X_i = torch.stack((mesh_x_i, mesh_t_i), dim=2).flatten(end_dim=1)
                X_i = X_i.to(torch.get_default_device())
                with torch.no_grad():
                    u_chunk = net(X_i).cpu().numpy().reshape(x.numel(), t_chunk.numel(),3)
                rho, rho_v, rho_e = euler.convert_model_output(u_chunk, gamma)
                u_pred_chunks.append(rho)
            usol = usol[:,:,0]
            pred = np.concatenate(u_pred_chunks, axis=1)

        else:
            x, t, usol = test_data.x,test_data.t,test_data.usol.numpy()
            loaded_net = models.FlexibleNet(width = 40, depth = 7, activation = nn.Tanh, input = 2, output = 3)
            loaded_net.load_state_dict(torch.load(NET_PATH))
            loaded_net.eval()
            mesh_x,mesh_t = torch.meshgrid(x.squeeze(),t.squeeze(),indexing="ij")
            X = torch.stack((mesh_x, mesh_t), dim=2)
            X = torch.flatten(X, end_dim=1)
            X = X.to(torch.get_default_device())
            with torch.no_grad():
                u_pred = loaded_net(X).cpu().numpy()

            u_pred = u_pred.reshape(x.numel(), t.numel(), 3)
            rho, rho_v, rho_e = euler.convert_model_output(u_pred, gamma)
            usol = usol[:,:,0]
            pred = rho
    else:
        if CAUSAL:
            x, t, usol = test_data.x,test_data.t,test_data.usol.cpu().numpy()
            networks = []
            for i in range(SLICES):
                loaded_net = models.FlexibleNet(width = 18, depth = 7, activation = nn.Tanh, input = 2, output = 1)
                loaded_net.load_state_dict(torch.load(NET_PATH/f"net_{i}.pth"))
                networks.append(loaded_net)

            mesh_x, mesh_t = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
            X = torch.stack((mesh_x, mesh_t), dim=2)
            X = torch.flatten(X, end_dim=1)
            X = X.to(torch.get_default_device())
            t_chunks = torch.chunk(t.squeeze(), SLICES)
            u_pred_chunks = []
            for net, t_chunk in zip(networks, t_chunks):
                mesh_x_i, mesh_t_i = torch.meshgrid(x.squeeze(), t_chunk, indexing="ij")
                X_i = torch.stack((mesh_x_i, mesh_t_i), dim=2).flatten(end_dim=1)
                X_i = X_i.to(torch.get_default_device())
                with torch.no_grad():
                    u_chunk = net(X_i).cpu().numpy().reshape(x.numel(), t_chunk.numel())
                u_pred_chunks.append(u_chunk)
            pred = np.concatenate(u_pred_chunks, axis=1)
        else:
            loaded_net = models.FlexibleNet(width = 26, depth = 4, activation = nn.Tanh, input = 2, output = 1)
            loaded_net.load_state_dict(torch.load(NET_PATH))

            loaded_net.eval()
            x, t, usol = test_data.x,test_data.t,test_data.usol.cpu().numpy()

            mesh_x,mesh_t = torch.meshgrid(x.squeeze(),t.squeeze(),indexing="ij")
            X = torch.stack((mesh_x, mesh_t), dim=2)
            X = torch.flatten(X, end_dim=1)
            X = X.to(torch.get_default_device())

            with torch.no_grad():
                pred = loaded_net(X).cpu().numpy()

            pred = pred.reshape(x.numel(), t.numel())



    generate_graph(pred=pred,x=x, t=t,usol=usol, save_path=save_path, filename=FILENAME)

    save_1d_output_graphs(save_path, usol, pred, mesh_x, mesh_t)

    calculate_relative_error(pred, usol)
