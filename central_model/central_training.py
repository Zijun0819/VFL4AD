import csv

from utils.training_utils import *
from DataPreProcessing.utils.dataprocessutils import *
import matplotlib.pyplot as plt
import pickle
from STGCN.stgcn import STGCN
import torch.nn as nn


def train_epoch(training_input, training_target, batch_size, net, optimizer, loss_criterion):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)


def model_train(A_wave, loss_criterion, train_input, train_target, val_input, val_target):
    net = STGCN(A_wave.shape[0], train_input.shape[3], InitParameters.NUM_STEPS_INPUT,
                InitParameters.NUM_STEPS_OUTPUT).to(device=args.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    training_losses = []
    validation_losses = []
    validation_maes = []
    density_maes = []
    flow_maes = []
    validation_rmses = []
    density_rmse = []
    flow_rmse = []

    for epoch in range(InitParameters.EPOCHS):
        loss = train_epoch(train_input, train_target,
                           batch_size=InitParameters.BATCH_SIZE, net=net, optimizer=optimizer, loss_criterion=loss_criterion)
        training_losses.append(loss)
        scheduler.step()

        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)

            out = net(A_wave, val_input)
            val_loss = loss_criterion(out, val_target).to(device="cpu")
            validation_losses.append(val_loss.detach().numpy().item())

            out_unnormalized = out.detach().cpu().numpy() * val_stds[0] + val_means[0]
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

        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print(f"Validation MAE: {validation_maes[-1]} \t density: {mae_density} \t flow: {mae_flow}")
        print(f"Validation RMSE: {validation_rmses[-1]} \t density: {rmse_density} \t flow: {rmse_flow}")
        if (epoch+1) % 500 == 0:
            plt.plot(training_losses, label="training loss")
            plt.plot(validation_losses, label="validation loss")
            plt.title("Results of centralized training")
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

    torch.save(net, model_save_pth)

    return net


if __name__ == '__main__':
    args = training_presetting()
    run_time = 1
    model_save_pth = f"checkpoints/central_model_{run_time}.pth"
    losses_save_pth = f"checkpoints/central_model_{run_time}.csv"

    train_input, train_target, val_input, val_target, A_wave, val_means, val_stds = get_central_dataset()
    train_target = torch.squeeze(train_target)
    val_target = torch.squeeze(val_target)
    test_input = val_input.to(device=args.device)
    test_target = val_target.to(device=args.device)

    loss_criterion = nn.MSELoss()
    A_wave = A_wave.to(device=args.device)

    if os.path.exists(model_save_pth):
        net = torch.load(model_save_pth)
    else:
        net = model_train(A_wave, loss_criterion, train_input, train_target, val_input, val_target)

    with torch.no_grad():
        net.eval()

        out = net(A_wave, test_input)
        test_loss = loss_criterion(out, test_target).to(device="cpu")
        print(f"The normalized loss of test data is: {test_loss.detach().numpy().item()}")

        out_unnormalized = out.detach().cpu().numpy() * val_stds[0] + val_means[0]
        target_unnormalized = test_target.detach().cpu().numpy() * val_stds[0] + val_means[0]
        mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
        rmse = np.sqrt(np.mean(np.square(out_unnormalized - target_unnormalized)))
        mae_density = np.mean(np.absolute(out_unnormalized[:, :, 0] - target_unnormalized[:, :, 0]))
        mae_flow = np.mean(np.absolute(out_unnormalized[:, :, 1] - target_unnormalized[:, :, 1]))
        rmse_density = np.sqrt(np.mean(np.square(out_unnormalized[:, :, 0] - target_unnormalized[:, :, 0])))
        rmse_flow = np.sqrt(np.mean(np.square(out_unnormalized[:, :, 1] - target_unnormalized[:, :, 1])))
        print(
            f"The mae and rmse in terms of density and flow are: {mae_density} \t {mae_flow} \t {rmse_density} \t {rmse_flow}, respectively")

