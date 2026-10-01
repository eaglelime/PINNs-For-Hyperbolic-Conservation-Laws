import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

def validate_net(network, test_data_loader, test_loss_fn):
    test_loss = 0
    with torch.no_grad():
        for X, usol in test_data_loader:
            X = X.to(torch.get_default_device())
            usol = usol.to(torch.get_default_device())
            pred = network(X).squeeze()
            test_loss = test_loss_fn(pred, usol)
    print(f"validation loss: {test_loss} \n")
    return test_loss

def train(test_data, net_trainer, epochs, print_freq, test_freq, early_stop_threshold, stop_epoch, test_loss_fn = nn.MSELoss()):
    test_data_loader = DataLoader(test_data,batch_size = len(test_data))
    for epoch in range(epochs):
        net_trainer.train(epoch, epoch % print_freq == 0)
        if epoch % test_freq == test_freq - 1:
            loss = validate_net(network = net_trainer.network, test_data_loader=test_data_loader, test_loss_fn = test_loss_fn)
        if epoch % stop_epoch == stop_epoch-1:
            loss = validate_net(network = net_trainer.network, test_data_loader=test_data_loader, test_loss_fn = test_loss_fn)
            if torch.isnan(loss) or loss > early_stop_threshold:
                print(f"Loss above threshod {early_stop_threshold}, aborting.")
                return current_loss, True
        if epoch % test_freq == 0:
            current_loss = validate_net(network = net_trainer.network, test_data_loader=test_data_loader, test_loss_fn = test_loss_fn)
        if epoch % print_freq == print_freq-1:
            print(f"Epoch: {epoch}")
            net_trainer.print_fn()
    net_trainer.print_fn()
    current_loss = validate_net(network = net_trainer.network, test_data_loader=test_data_loader, test_loss_fn = test_loss_fn)
    return current_loss, False, net_trainer.network

def min_max_train(test_data, net_trainer, adv_trainers, epochs, print_freq, test_freq, early_stop_threshold, stop_epoch, test_loss_fn = nn.MSELoss()):
    test_data_loader = DataLoader(test_data,batch_size = len(test_data))
    sequence_validation_loss = None
    losses = []
    training_losses = []
    for epoch in range(epochs):
        for adv_trainer in adv_trainers:
            adv_trainer.train(epoch, epoch % print_freq == 0)
        training_loss = net_trainer.train(epoch, epoch % print_freq == 0)
        if epoch%1==0:
            training_losses.append((epoch,training_loss))
        if epoch % test_freq == test_freq - 1:
            loss = validate_net(network = net_trainer.network, test_data_loader=test_data_loader, test_loss_fn = test_loss_fn)
            losses.append((epoch, loss))
        if epoch % stop_epoch == stop_epoch-1:
            loss = validate_net(network = net_trainer.network, test_data_loader=test_data_loader, test_loss_fn = test_loss_fn)
            if torch.isnan(loss) or loss > early_stop_threshold:
                print(f"Loss above threshod {early_stop_threshold}, aborting.")
                net_loss = validate_net(network = net_trainer.network, test_data_loader=test_data_loader, test_loss_fn = test_loss_fn)
                return losses,training_losses, True, net_trainer.network, net_loss
        if epoch % print_freq == print_freq-1:
            print(f"Epoch: {epoch}")
            net_trainer.print_fn()
            for adv_trainer in adv_trainers:
                adv_trainer.print_fn()
    net_trainer.print_fn()
    for adv_trainer in adv_trainers:
        adv_trainer.print_fn()
    net_loss = validate_net(network = net_trainer.network, test_data_loader=test_data_loader, test_loss_fn = test_loss_fn)
    return losses, training_losses, False, net_trainer.network, net_loss
