import numpy as np
from pathlib import Path
from scipy.io import savemat

N0 = 1000
BC_num = 100
x_size = 512
t_size = 400
C = 2
x_ub = 1
x_lb = -1
t_ub = 0.99
t_lb = 0

parent_path= Path(__file__).resolve().parent.parent
DATA_PATH = parent_path/ "data" / "linear_advec_data.mat" 


def pde(x,t, init_cond):
    x_mesh,t_mesh = np.meshgrid(x,t,indexing="ij")
    return init_cond(x_mesh-C*t_mesh)


if __name__ == "__main__":
    x = np.linspace(x_lb,x_ub,x_size)
    t=np.linspace(t_lb,t_ub,t_size)

    x0 = np.linspace(x_lb,x_ub,N0)

    t_bc = np.linspace(t_lb,t_ub,BC_num)

    init_cond = lambda x : np.heaviside(np.sin(np.pi*x),0)
    usol = pde(x.T,t,init_cond)
    usol0 = pde(x0.T,0,init_cond)

    boundaries = {"x_ub":x_ub,
                  "x_lb":x_lb,
                  "t_ub":t_ub,
                  "t_lb":t_lb,
                  "x_size":x_size,
                  "t_size":t_size}
    data_dict = {"x0":x0,
                "x":x,
                "t":t,
                "t_bc":t_bc,
                "usol0":usol0,
                "usol":usol,
                "boundaries":boundaries}

    savemat(DATA_PATH, data_dict)


