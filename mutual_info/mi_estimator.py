import torch
import torch.optim as optim
import numpy as np
# from DataPreProcessing.utils.dataprocessutils import extract_data4vfl_2
from mutual_info.mine import MINE
from InitParas import InitParameters


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = torch.tensor(joint, dtype=torch.float32, device='cuda')
    marginal = torch.tensor(marginal, dtype=torch.float32, device='cuda')
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate)*ma_et + ma_rate*torch.mean(et)

    # unbiasing use moving average
    loss = -(torch.mean(t) - (1 / ma_et.mean()).detach()*torch.mean(et))
    # use biased estimator
    #     loss = - mi_lb

    mine_net_optim.zero_grad()
    loss.backward()
    mine_net_optim.step()
    return mi_lb, ma_et


def sample_batch(data, x_dim, batch_size=100, sample_mode='joint'):
    if sample_mode == 'joint':
        index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[index]
    else:
        joint_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = np.concatenate([data[joint_index][:, :x_dim], data[marginal_index][:, x_dim:]], axis=1)
    return batch


def mi_train(data, mine_net, mine_net_optim, x_dim, batch_size=100, iter_num=int(5e+3), log_freq=int(1e+3)):
    result = list()
    ma_et = 1.
    for i in range(iter_num):
        batch = sample_batch(data, x_dim, batch_size=batch_size), sample_batch(data, x_dim, batch_size=batch_size, sample_mode='marginal')
        mi_lb, ma_et = learn_mine(batch, mine_net, mine_net_optim, ma_et)
        result.append(mi_lb.detach().cpu().numpy())
        if (i+1) % (log_freq) == 0:
            print(result[-1])
    return result


def mi_data_processing(x_data, y_data, sample_count):
    x_data = x_data.transpose(0, 1, 3, 2)
    x_dim = np.prod(x_data.shape[1:]).item()
    train_features = x_data.reshape(x_data.shape[0], -1)
    train_target = y_data.reshape(y_data.shape[0], -1)
    data = np.concatenate((train_features, train_target), axis=1)
    _data_joint = sample_batch(data, x_dim, sample_count)
    _data_marginal = sample_batch(data, x_dim, sample_count, 'marginal')
    _data_joint = torch.tensor(_data_joint, dtype=torch.float32, device='cuda')
    _data_marginal = torch.tensor(_data_marginal, dtype=torch.float32, device='cuda')

    return _data_joint, _data_marginal


if __name__ == '__main__':
    print("==========Data is processing!==========")
    # features, target, _, _ = extract_data4vfl_2(indice_begin=0, indice_end=len(InitParameters.DATA_LIST),
    #                                             count_parties=InitParameters.NUM_PASSIVE_CLIENTS)
    # features = [f.transpose(0, 1, 3, 2) for f in features]
    # x_dim = np.prod(features[0].shape[1:]).item()
    # y_dim = np.prod(target.shape[1:]).item()
    # mi_est_1 = MINE(input_size=x_dim+y_dim, hidden_size=100).to(device='cuda')
    # mi_est_1_optim = optim.Adam(mi_est_1.parameters(), lr=1e-3, weight_decay=5e-3)
    # train_features = [f.reshape(f.shape[0], -1)[:300] for f in features]
    # train_target = target.reshape(target.shape[0], -1)[:300]
    # data = np.concatenate((train_features[0], train_target), axis=1)
    # mi_train(data=data, mine_net=mi_est_1, mine_net_optim=mi_est_1_optim, x_dim=x_dim)
