from InitParas.initparas import InitParameters
from mutual_info.mi_estimator import mutual_information, mi_data_processing
from mutual_info.mine import MINE
from .roadnetutils import load_processed_data
import math
import os
import sys
import random
import torch
import numpy as np


current_dir = os.path.dirname(__file__)  # 获取当前脚本的目录
project_root = os.path.dirname(current_dir)  # 上一层目录
project_root = os.path.dirname(project_root)
mine_module_path = os.path.join(project_root, 'mutual_info')

# 将 mine 模块的路径添加到 sys.path
sys.path.append(mine_module_path)


def generate_dataset(X, features, target, *, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, :, i + num_timesteps_input: j])


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A.astype(np.float32)
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def normalize_data(X, X_mean, X_std):
    # Normalization using Z-score method
    X = X - X_mean.reshape(1, -1, 1)
    X = X / X_std.reshape(1, -1, 1)

    return X


def normalize_selected_data(data_lists, means, stds):
    features, target = [], []

    for data in data_lists:
        normalized_density_flow = normalize_data(data, means, stds)

        generate_dataset(normalized_density_flow, features, target,
                         num_timesteps_input=InitParameters.NUM_STEPS_INPUT,
                         num_timesteps_output=InitParameters.NUM_STEPS_OUTPUT)

    return np.array(features), np.array(target)


def get_normalized_paras():
    data_lists = []
    for data_train in InitParameters.DATA_LIST:
        file_path = os.path.join('..//Data', data_train.data_sub_folder,
                                 f'{data_train.get_time_identifier}_features.pkl')
        density_flow = load_processed_data(file_path)
        data_lists.append(density_flow)

    all_data = np.concatenate(data_lists, axis=2)
    all_data = all_data.astype(np.float32)
    means = np.mean(all_data, axis=(0, 2))
    stds = np.std(all_data, axis=(0, 2))

    return means, stds


def split_train_val_test_data(indice_begin, indice_end):
    data_list = []

    for i in range(indice_begin, indice_end):
        data_train = InitParameters.DATA_LIST[i]
        file_path = os.path.join('..//Data', data_train.data_sub_folder, f'{data_train.get_time_identifier}_features.pkl')
        density_flow = load_processed_data(file_path)
        data_list.append(density_flow)

    features, target, means, stds = dataset_generation_separate_normalized(data_list)

    return features, target, means, stds


def dataset_generation_separate_normalized(data_list):
    all_data = np.concatenate(data_list, axis=2)
    all_data = all_data.astype(np.float32)
    means = np.mean(all_data, axis=(0, 2))
    stds = np.std(all_data, axis=(0, 2))

    features, target = normalize_selected_data(data_list, means, stds)

    return features, target, means, stds


def dataset_generation_all_normalized(data_list):
    means, stds = get_normalized_paras()

    features, target = normalize_selected_data(data_list, means, stds)

    return features, target, means, stds


# TODO: This function need to be revised, so as to make it to be more generalizable
def extract_data4vfl(indice_begin, indice_end, count_parties):
    data_lists = [[] for _ in range(count_parties)]
    road_count = len(InitParameters.SELECTED_ROAD_PATHS)
    if count_parties == 2:
        print("The process of data preprocessing of the test demon for the vfl training")

    indices_one = np.random.choice(road_count, 13, replace=False)
    # The indices that do not be selected by the first party will set as the indices of the second party
    indices_two = np.setdiff1d(np.arange(road_count), indices_one)

    for i in range(indice_begin, indice_end):
        data_train = InitParameters.DATA_LIST[i]
        file_path = os.path.join('..//Data', data_train.data_sub_folder, f'{data_train.get_time_identifier}_features.pkl')
        density_flow = load_processed_data(file_path)

        # Extract the data as per the indices of each party and pad the data of each party with 0
        party_one_data = np.zeros_like(density_flow)
        party_two_data = np.zeros_like(density_flow)

        party_one_data[indices_one, :, :] = density_flow[indices_one, :, :]
        party_two_data[indices_two, :, :] = density_flow[indices_two, :, :]
        data_lists[0].append(party_one_data)
        data_lists[1].append(party_two_data)

    _, central_target, central_means, central_stds = split_train_val_test_data(indice_begin, indice_end)

    features_lists = []

    for data_list in data_lists:
        features, _, _, _ = dataset_generation_separate_normalized(data_list)
        features_lists.append(features)

    return features_lists, central_target, central_means, central_stds


# The generalized version of the function extract_data4vfl()
def extract_data4vfl_(indice_begin, indice_end, count_parties, malicious_exist=True):
    data_lists = [[] for _ in range(count_parties)]
    road_count = len(InitParameters.SELECTED_ROAD_PATHS)
    random.seed()
    np.random.seed()
    road_cnts = random.choices(range(InitParameters.ROAD_MONITOR_MIN_, InitParameters.ROAD_MONITOR_MAX_+1), k=count_parties)

    road_indices_parties = [np.random.choice(road_count, road_cnt, replace=False) for road_cnt in road_cnts]

    for i in range(indice_begin, indice_end):
        data_train = InitParameters.DATA_LIST[i]
        file_path = os.path.join('..//Data', data_train.data_sub_folder, f'{data_train.get_time_identifier}_features.pkl')
        density_flow = load_processed_data(file_path)

        # Extract the data as per the indices of each party and pad the data of each party with 0
        partys_data = [np.zeros_like(density_flow) for _ in range(count_parties)]
        for j, road_index in enumerate(road_indices_parties):
            partys_data[j][road_index, :, :] = density_flow[road_index, :, :]
            data_lists[j].append(partys_data[j])

    _, central_target, central_means, central_stds = split_train_val_test_data(indice_begin, indice_end)

    features_lists = []

    for data_list in data_lists:
        features, _, _, _ = dataset_generation_separate_normalized(data_list)
        features_lists.append(features)

    return features_lists, central_target, central_means, central_stds


def extract_data4vfl_2(indice_begin, indice_end, malicious_exist=False, malicious_flag=0, noise_mean=0, noise_std=1, num_mal=1, lp=0.1):
    np.random.seed(51)
    random.seed(51)
    count_parties = InitParameters.NUM_PASSIVE_CLIENTS
    data_lists = [[] for _ in range(count_parties)]
    road_count = len(InitParameters.SELECTED_ROAD_PATHS)

    road_cnts = random.choices(range(InitParameters.ROAD_MONITOR_MIN, InitParameters.ROAD_MONITOR_MAX+1), k=(count_parties-1))
    road_cnts.append(road_count - sum(road_cnts))
    available_indices = np.arange(0, road_count)

    # 存储每一次选择的结果
    road_indices_parties = []

    # 遍历每次需要选择的数量
    for road_cnt in road_cnts:
        # 从当前可用的索引中随机选择指定数量的索引
        selected_indices = np.random.choice(available_indices, road_cnt, replace=False)
        # 将本次选择的索引保存
        road_indices_parties.append(selected_indices)
        # 更新可用索引，移除已经被选择的
        available_indices = np.setdiff1d(available_indices, selected_indices)

    for i in range(indice_begin, indice_end):
        data_train = InitParameters.DATA_LIST[i]
        file_path = os.path.join('..//Data', data_train.data_sub_folder, f'{data_train.get_time_identifier}_features.pkl')
        density_flow = load_processed_data(file_path)

        # Extract the data as per the indices of each party and pad the data of each party with 0
        partys_data = [np.zeros_like(density_flow) for _ in range(count_parties)]
        for j, road_index in enumerate(road_indices_parties):
            partys_data[j][road_index, :, :] = density_flow[road_index, :, :]
            data_lists[j].append(partys_data[j])

    _, central_target, central_means, central_stds = split_train_val_test_data(indice_begin, indice_end)

    features_lists = []
    # TODO: Randomly opt an index as dishonest node, the data provided by this party will possess certain noise
    lazy_node_index = [random.randint(0, count_parties-1) for _ in range(num_mal)]
    if num_mal == count_parties:
        lazy_node_index = [i for i in range(count_parties)]
    for mp_index, data_list in enumerate(data_lists):
        features, _, _, _ = dataset_generation_separate_normalized(data_list)
        if mp_index in lazy_node_index and malicious_exist:
        # if malicious_exist:
            features = add_noise_to_features(features, malicious_flag, noise_mean, noise_std, lp)

        features_lists.append(features)

    return features_lists, central_target, central_means, central_stds


def add_noise_to_features(features, flag=0, noise_mean=0, noise_std=1, lazy_percentage=0.1):
    noise_added_features = None
    historical_data = features[:300, :, :, :]
    data_len = features.shape[0]
    num_lazy = int(data_len * lazy_percentage)

    # Randomly select num_lazy indices
    lazy_indices = np.random.choice(data_len, num_lazy, replace=False)

    random_historical_indices = np.random.choice(historical_data.shape[0], size=num_lazy, replace=True)

    # 从 historical_data 中根据生成的随机索引选择数据
    random_historical_data = historical_data[random_historical_indices]

    if flag == 1:
        noise_added_features = features + np.random.normal(noise_mean, noise_std, features.shape).astype(np.float32)
    elif flag == 2:
        random_features = np.random.normal(noise_mean, noise_std, features.shape).astype(np.float32)
        features[lazy_indices] = random_features[lazy_indices]

        noise_added_features = features
    elif flag == 3:
        noise_added_features = np.roll(features, shift=100, axis=0).astype(np.float32)
        noise_added_features[:100, :, :, :] = features[-100:, :, :, :]
        features[lazy_indices] = random_historical_data

        noise_added_features = features

    return noise_added_features


def get_central_dataset():
    random.shuffle(InitParameters.DATA_LIST)
    train_features, train_target, _, _ = split_train_val_test_data(0, math.ceil(len(InitParameters.DATA_LIST) * 0.8))
    val_features, val_target, val_means, val_stds = split_train_val_test_data(
        math.ceil(len(InitParameters.DATA_LIST) * 0.8), len(InitParameters.DATA_LIST))

    normalized_A = get_normalized_adj(InitParameters.TRAFFIC_GRAPH_A)
    A_wave = torch.from_numpy(normalized_A)

    return torch.from_numpy(train_features), torch.from_numpy(train_target), torch.from_numpy(
        val_features), torch.from_numpy(np.array(val_target)), A_wave, val_means, val_stds


def vfl_data_prepare(data_begin, data_end):
    noise_seq = np.linspace(0, 0.3, 4)
    vfl_data = []
    mean_avg = []
    std_avg = []
    for noise in noise_seq:
        t_features, t_labels, m, s = extract_data4vfl_2(data_begin, data_end, malicious_exist=True, malicious_flag=1,
                                                        noise_mean=noise, noise_std=noise)
        vfl_data.append(t_features)
        mean_avg.append(m)
        std_avg.append(s)

    t_lazy_1_features, t_labels, m, s = extract_data4vfl_2(data_begin, data_end, malicious_exist=True, malicious_flag=2,
                                                           noise_mean=0, noise_std=1)
    vfl_data.append(t_lazy_1_features)
    mean_avg.append(m)
    std_avg.append(s)

    t_lazy_2_features, t_labels, m, s = extract_data4vfl_2(data_begin, data_end, malicious_exist=True, malicious_flag=3)
    vfl_data.append(t_lazy_2_features)
    mean_avg.append(m)
    std_avg.append(s)

    vfl_road_data = []
    for i in range(len(t_lazy_2_features)):
        data_road_i = [vfl[i] for vfl in vfl_data]
        vfl_road_data.append(data_road_i)

    mean = np.mean(np.array(mean_avg, dtype=np.float32), axis=0)
    std = np.mean(np.array(std_avg, dtype=np.float32), axis=0)

    return vfl_road_data, t_labels, mean, std


def identify_vfl_clients_by_mi(data_begin, data_end):
    vfl_road_data, t_labels, _, _ = vfl_data_prepare(data_begin, data_end)
    vfl_train_data = []

    for model_road_index in range(len(vfl_road_data)):
        mi_res = []
        for road_data in vfl_road_data[model_road_index]:
            mi_model_save_pth = f"..\\checkpoints\\mi_est_{model_road_index + 1}.pth"
            mi_est = torch.load(mi_model_save_pth, weights_only=False)
            data_joint, data_marginal = mi_data_processing(road_data, t_labels, sample_count=InitParameters.MI_SAMPLE_CNT)
            mi_out, _, _ = mutual_information(data_joint, data_marginal, mi_est)

            mi_res.append(mi_out.detach().cpu().item())
        # Select the index with the maximum mi value as the vfl model training data
        max_mi_index = mi_res.index(max(mi_res))
        vfl_train_data.append(vfl_road_data[model_road_index][max_mi_index])

    return vfl_train_data, t_labels


def random_identify_vfl_clients(data_begin, data_end):
    vfl_road_data, t_labels, _, _ = vfl_data_prepare(data_begin, data_end)
    vfl_train_data = []
    random.seed(51)
    for model_road_index in range(len(vfl_road_data)):
        # Randomly select the index of vfl model training data
        vfl_data_index = random.randint(0, len(vfl_road_data[model_road_index])-1)
        vfl_train_data.append(vfl_road_data[model_road_index][vfl_data_index])

    return vfl_train_data, t_labels


def identify_vfl_clients_by_oracle(data_begin, data_end):
    vfl_road_data, t_labels, _, _ = vfl_data_prepare(data_begin, data_end)
    vfl_train_data = []
    for model_road_index in range(len(vfl_road_data)):
        # The vfl model training data is identified by oracle
        vfl_data_index = 0
        vfl_train_data.append(vfl_road_data[model_road_index][vfl_data_index])

    return vfl_train_data, t_labels


def identify_vfl_clients_by_vflfs(data_begin, data_end):
    vfl_road_data, t_labels, mean, std = vfl_data_prepare(data_begin, data_end)
    vfl_train_data = []
    for model_road_index in range(len(vfl_road_data)):
        # The vfl model training data is identified by oracle
        for vfl_data_index in range(len(vfl_road_data[model_road_index])):
            vfl_train_data.append(vfl_road_data[model_road_index][vfl_data_index])

    return vfl_train_data, t_labels, mean, std


def get_vfl_dataset(flag=1, malicious=False, mal_flag=3, num_mal=1):
    random.shuffle(InitParameters.DATA_LIST)
    train_index_begin = 0
    train_index_end = math.ceil(len(InitParameters.DATA_LIST) * 0.8)

    val_features_lists, val_target, val_means, val_stds = extract_data4vfl_2(
        math.ceil(len(InitParameters.DATA_LIST) * 0.8), len(InitParameters.DATA_LIST), malicious_exist=malicious,
        malicious_flag=mal_flag, num_mal=num_mal, lp=1)

    if flag == 1:
        train_features_lists, train_target = identify_vfl_clients_by_mi(train_index_begin, train_index_end)
    elif flag == 2:
        train_features_lists, train_target = random_identify_vfl_clients(train_index_begin, train_index_end)
    elif flag == 3:
        train_features_lists, train_target = identify_vfl_clients_by_oracle(train_index_begin, train_index_end)
    elif flag == 4:
        train_features_lists, train_target, _, _ = identify_vfl_clients_by_vflfs(train_index_begin, train_index_end)
        val_features_lists, val_target, val_means, val_stds = identify_vfl_clients_by_vflfs(
            math.ceil(len(InitParameters.DATA_LIST) * 0.8), len(InitParameters.DATA_LIST))

    normalized_A = get_normalized_adj(InitParameters.TRAFFIC_GRAPH_A)

    A_wave = torch.from_numpy(normalized_A)

    return train_features_lists, torch.from_numpy(train_target), val_features_lists, torch.from_numpy(
        val_target), A_wave, val_means, val_stds
