
from operator import itemgetter
import torch
import matplotlib.pyplot as plt
import numpy as np

def kl_annealing(info,type="cyclical"):
    """
    Sources: https://medium.com/@ragy202/addressing-posterior-collapse-in-chemical-vaes-151c0f210388

    T = Total steps
    M = Number of cycles
    R = Proportion used to increase beta
    t = Global step

    """
    step,total_step = itemgetter("step","total_step")(info)
    max_kl_weight = 1
    k=0.000005 #lower values smoother logistic regression
    #T = 150000 #higher values, less cycles
    #T = 150000 #higher values, less cycles
    T = total_step #higher values, less cycles
    R = 0.5  # higher values, the max_kl_weight is maintained for a shorter time
    M = 4  # number of cycles - 1

    if type == "cyclical": #https://arxiv.org/pdf/1903.10145
        period = (T / M)
        internal_period = (step) % (period)  # Itteration_number/(Global Period)
        beta = internal_period / period
        if beta > R:
            beta = max_kl_weight
        else:
            beta = min(max_kl_weight, beta / R)  # Linear function
        return beta
    elif type == "linear":
        return min(max_kl_weight, step / T*0.3)
    elif type == "logistic":
        return float(max_kl_weight / (1 + np.exp(-k * (step - T))))


def kl_annealing_torch(info,type="cyclical"):
    """
    Sources: https://medium.com/@ragy202/addressing-posterior-collapse-in-chemical-vaes-151c0f210388

    M = Number of cycles
    R = Proportion used to increase beta, higher values, the max_kl_weight is maintained for a shorter time
    t = Global step


    """
    step,total_step = itemgetter("step","total_step")(info)
    max_kl_weight = 1.
    k= torch.tensor([0.00005])
    T = torch.tensor([total_step])
    R = torch.tensor([0.5])  # higher values, the max_kl_weight is maintained for a shorter time
    M = torch.tensor([4])  # number of cycles - 1

    if type == "cyclical": #https://arxiv.org/pdf/1903.10145
        period = (T / M)
        internal_period = (step) % (period)  # Itteration_number/(Global Period)
        beta = internal_period / period
        if beta > R:
            beta = max_kl_weight
        else:
            beta = min(max_kl_weight, beta / R)  # Linear function
        return torch.Tensor([beta]).numpy()
    elif type == "linear":
        return torch.Tensor([min(max_kl_weight, step / T)])
    elif type == "logistic":
        return torch.Tensor([float(max_kl_weight / (1 + torch.exp(-k * (step - T))))]).numpy()
    elif type == None:
        return torch.tensor([max_kl_weight]).numpy()



x,y_cyclical,y_lin,y_log = [],[],[],[]
num_epochs = 2000
n_train_seqs = 200
batch_size = 1
total_step = num_epochs * (n_train_seqs / batch_size)

info = dict(step=1,
            total_step = total_step,
            )

for i in range(int(total_step)):
    info["step"] = i
    x.append(i)
    y_cyclical.append(kl_annealing(info,type="cyclical"))
    y_lin.append(kl_annealing(info,type="linear"))
    y_log.append(kl_annealing(info,type="logistic"))

fig, axs = plt.subplots(nrows=1,ncols=3)

axs[0].plot(x,y_cyclical)
axs[0].set_title("Cyclical")

axs[1].plot(x,y_lin)
axs[1].set_title("Monotonic/Linear")

axs[2].plot(x,y_log)
axs[2].set_title("Logistic")


plt.show()
