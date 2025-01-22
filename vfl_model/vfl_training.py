import csv
import time
import os
from codecarbon import EmissionsTracker
from utils.training_utils import *
from torch import nn
import torch

from vfl_model.mp_client import PassiveClient
from vfl_model.ma_client import ActiveClient
import numpy as np

from vfl_model.splitNN import SplitNN
from InitParas import InitParameters
from DataPreProcessing.utils.dataprocessutils import get_vfl_dataset
from matplotlib import pyplot as plt


def vfl_train_epoch(training_inputs, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_inputs: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_inputs[0].shape[0])

    epoch_training_losses = []
    for i in range(0, training_inputs[0].shape[0], batch_size):
        vfl_net.train()
        vfl_net.zero_grads()

        indices = permutation[i:i + batch_size]
        X_batches = [training_input[indices].to(device=args.device) for training_input in training_inputs]
        y_batch = training_target[indices].to(device=args.device)

        vfl_out = vfl_net(X_batches)
        vfl_loss = loss_criterion(vfl_out, y_batch)
        vfl_loss.backward()
        vfl_net.backward()
        vfl_net.step()

        epoch_training_losses.append(vfl_loss.detach().cpu().numpy())

    return sum(epoch_training_losses) / len(epoch_training_losses)


def vfl_model_train(train_inputs, train_target, val_inputs, val_target):
    training_losses = []
    validation_losses = []
    validation_maes = []
    density_maes = []
    flow_maes = []
    validation_rmses = []
    density_rmse = []
    flow_rmse = []

    for epoch in range(InitParameters.EPOCHS):
        epoch_loss = vfl_train_epoch(train_inputs, train_target, batch_size=InitParameters.BATCH_SIZE)
        training_losses.append(epoch_loss)
        vfl_net.learning_rate_decay()

        # Run validation
        with torch.no_grad():
            vfl_net.eval()
            val_inputs = [val_input.to(device=args.device) for val_input in val_inputs]
            val_target = val_target.to(device=args.device)

            validation_out = vfl_net(val_inputs)
            val_loss = loss_criterion(validation_out, val_target).to(device="cpu")
            validation_losses.append(val_loss.detach().numpy().item())

            out_unnormalized = validation_out.detach().cpu().numpy() * val_stds[0] + val_means[0]
            target_unnormalized = val_target.detach().cpu().numpy() * val_stds[0] + val_means[0]
            mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
            mae_density = np.mean(np.absolute(out_unnormalized[:, :, 0] - target_unnormalized[:, :, 0]))
            mae_flow = np.mean(np.absolute(out_unnormalized[:, :, 1] - target_unnormalized[:, :, 1]))
            rmse = np.sqrt(np.mean(np.square(out_unnormalized - target_unnormalized)))
            rmse_density = np.sqrt(np.mean(np.square(out_unnormalized[:, :, 0] - target_unnormalized[:, :, 0])))
            rmse_flow = np.sqrt(np.mean(np.square(out_unnormalized[:, :, 1] - target_unnormalized[:, :, 1])))
            validation_maes.append(mae)
            density_maes.append(mae_density)
            flow_maes.append(mae_flow)
            density_rmse.append(rmse_density)
            flow_rmse.append(rmse_flow)
            validation_rmses.append(rmse)

            # validation_out = None
            # val_inputs = [val_input.to(device="cpu") for val_input in val_inputs]
            # val_target = val_target.to(device="cpu")

        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print(f"Validation MAE: {validation_maes[-1]} \t density: {mae_density} \t flow: {mae_flow}")
        print(f"Validation RMSE: {validation_rmses[-1]} \t density: {rmse_density} \t flow: {rmse_flow}")
        if (epoch + 1) % 500 == 0:
            y_min, y_max = plt.ylim()
            plt.yticks(np.arange(y_min, y_max, 0.1))
            plt.plot(training_losses, label="training loss")
            plt.plot(validation_losses, label="validation loss")
            plt.title("Results of training with VFL (w/o data processing)")
            plt.legend()
            plt.show()

    checkpoint_path = "checkpoints/"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    losses = []
    for train_loss, val_loss, val_mae, density_1, flow_1, val_rmse, density_2, flow_2 in zip(training_losses,
                                                                                             validation_losses,
                                                                                             validation_maes,
                                                                                             density_maes, flow_maes,
                                                                                             validation_rmses,
                                                                                             density_rmse, flow_rmse):
        epoch_loss = [train_loss, val_loss, val_mae, density_1, flow_1, val_rmse, density_2, flow_2]
        losses.append(epoch_loss)

    with open(losses_save_pth, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(losses)

    torch.save(vfl_net, model_save_pth)

    return vfl_net


if __name__ == '__main__':
    args = training_presetting()

    loss_criterion = nn.MSELoss()
    # flag is in {1, 2, 3, 4}, which are represent MI, random, oracle, and vflfs, respectively.
    flag = 1
    run_time = 1

    model_save_pth = f"checkpoints/vfl_model_{run_time}_flag_{flag}.pth"
    losses_save_pth = f"checkpoints/vfl_losses_{run_time}_flag_{flag}.csv"

    t_inputs, t_target, v_inputs, v_target, A_wave, val_means, val_stds = get_vfl_dataset(flag=flag, malicious=True, num_mal=2)

    tensor_t_inputs = [torch.from_numpy(t_input) for t_input in t_inputs]
    tensor_v_inputs = [torch.from_numpy(v_input) for v_input in v_inputs]
    train_target = torch.squeeze(t_target)
    val_target = torch.squeeze(v_target)

    A_wave = A_wave.to(device=args.device)

    if os.path.exists(model_save_pth):
        vfl_net = torch.load(model_save_pth)
    else:
        pcs = [PassiveClient() for _ in range(InitParameters.NUM_PASSIVE_CLIENTS)]

        label_client = ActiveClient(InitParameters.NUM_FEATURES, InitParameters.NUM_PASSIVE_CLIENTS)

        vfl_net = SplitNN(A_wave, pcs, label_client)
        vfl_net.model_to_device(device=args.device)
        vfl_model_train(tensor_t_inputs, train_target, tensor_v_inputs, val_target)

    test_input = [tensor_v_in.to(device=args.device) for tensor_v_in in tensor_v_inputs]
    test_target = val_target.to(device=args.device)

    with torch.no_grad():
        # Model inference
        vfl_net.eval()
        out = vfl_net(test_input)

        test_loss = loss_criterion(out, test_target).to(device="cpu")
        print(f"The normalized loss of test data is: {test_loss.detach().numpy().item()}")

        out_unnormalized = out.detach().cpu().numpy() * val_stds[0] + val_means[0]
        target_unnormalized = test_target.detach().cpu().numpy() * val_stds[0] + val_means[0]
        mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        mae_times = np.mean(np.absolute(out_unnormalized - target_unnormalized), axis=(1, 2))
        print(f'The averaged mae of the traffic state is {mae:0.2f}')
        rmse = np.sqrt(np.mean(np.square(out_unnormalized - target_unnormalized)))
        mae_density = np.mean(np.absolute(out_unnormalized[:, :, 0] - target_unnormalized[:, :, 0]))
        mae_flow = np.mean(np.absolute(out_unnormalized[:, :, 1] - target_unnormalized[:, :, 1]))
        rmse_density = np.sqrt(np.mean(np.square(out_unnormalized[:, :, 0] - target_unnormalized[:, :, 0])))
        rmse_flow = np.sqrt(np.mean(np.square(out_unnormalized[:, :, 1] - target_unnormalized[:, :, 1])))
        print(f"The mae and rmse in terms of density and flow are: {mae_density} \t {mae_flow} \t {rmse_density} \t {rmse_flow}, respectively")
        out = None

    # TSE errors comparison code
    np.save('historical_data_res.npy', mae_times)
    random_data_res = np.load('random_data_res.npy')
    np.savetxt("random_data_res.csv", np.round(random_data_res, 2), delimiter=",", fmt='%.2f')
    normal_data_res = np.load('normal_data_res.npy')
    np.savetxt("normal_data_res.csv", np.round(normal_data_res, 2), delimiter=",", fmt='%.2f')
    historical_data_res = np.load('historical_data_res.npy')
    np.savetxt("historical_data_res.csv", np.round(historical_data_res, 2), delimiter=",", fmt='%.2f')
    y_min, y_max = plt.ylim()
    plt.yticks(np.arange(y_min, y_max, 0.1))
    plt.plot(normal_data_res[:50], label="normal data")
    plt.plot(random_data_res[:50], label="random data")
    plt.plot(historical_data_res[:50], label="historical data")
    plt.title("Comparison results of traffic state estimation with VFL")
    plt.xlabel("Time stamp")
    plt.ylabel("Predicting traffic state error")
    plt.legend()
    plt.show()
