import itertools
import pickle
import numpy as np
from DataPreProcessing.pneumadataset import PNeumaDataset
import osmnx as ox
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
import matplotlib.pyplot as plt
from InitParas.initparas import InitParameters

'''
The method in this package is utilized to extract feature information form the trajectories of vehicles on the road net
'''
def build_matching_graph(road_net_graph):
    map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)

    nodes_id = list(road_net_graph.nodes)
    for node in nodes_id:
        lat = road_net_graph.nodes[node]['y']
        lon = road_net_graph.nodes[node]['x']
        map_con.add_node(node, (lat, lon))

    edges_id = list(road_net_graph.edges)
    for edge in edges_id:
        node_a, node_b = edge[0], edge[1]
        map_con.add_edge(node_a, node_b)
        map_con.add_edge(node_b, node_a)  # Add reverse edge

    map_con.purge()

    return map_con


def map_path_to_graph(map_con, vehicle_loc):
    matcher = DistanceMatcher(map_con, max_dist=100,  # meter
                              min_prob_norm=0.001,
                              non_emitting_length_factor=0.75,
                              obs_noise=50, obs_noise_ne=75,  # meter
                              dist_noise=50,  # meter
                              non_emitting_states=True, only_edges=True)

    states, _ = matcher.match(vehicle_loc)
    nodes = matcher.path_pred_onlynodes

    return states, nodes


def grasp_road_infos(graph):
    road_infos = {}
    for dictionary in InitParameters.SELECTED_ROAD_PATHS:
        for key, value in dictionary.items():
            node_a_index = int(value[0])
            node_b_index = int(value[1])
            road_info = {}
            road_direction = True if graph.has_edge(node_a_index, node_b_index) else False
            if road_direction:
                road_length = int(graph.edges[node_a_index, node_b_index, 0]['length'])
                road_nodes = (node_a_index, node_b_index)
            else:
                road_length = int(graph.edges[node_b_index, node_a_index, 0]['length'])
                road_nodes = (node_b_index, node_a_index)
            road_info['length'] = road_length
            road_info['nodes'] = road_nodes

            road_infos[key] = road_info

    return road_infos


def plot_matched_edges_on_graph(graph, matched_traj):
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_color='r', node_size=5)

    # Extract and plot the matched edges
    for node_a_index, node_b_index in matched_traj:  # Assuming the first two columns hold the matched edge node indices
        node_a_index = int(node_a_index)  # Convert to integer as the indices are stored as floats in the array
        node_b_index = int(node_b_index)  # Convert to integer as the indices are stored as floats in the array

        if graph.has_edge(node_a_index, node_b_index):  # Check if the graph has this edge
            point_a = graph.nodes[node_a_index]
            point_b = graph.nodes[node_b_index]
            ax.plot([point_a['x'], point_b['x']], [point_a['y'], point_b['y']], color='blue', linewidth=2)

    plt.show()


def visualize_selected_paths(road_net_graph):
    road_infos = grasp_road_infos(road_net_graph)
    road_paths = [v['nodes'] for v in road_infos.values()]
    plot_matched_edges_on_graph(road_net_graph, road_paths)


def find_road_id(road_infos, target):
    if len(target) > 0:
        target = target[0]
        reversed_target = (target[1], target[0])
        for k, v in road_infos.items():
            if v['nodes'] == target or v['nodes'] == reversed_target:
                return k
        return None  # 如果没有找到，则返回None
    else:
        return None


def roadnet_data_processing(dataset_ids: list):
    selected_areas_tracks = PNeumaDataset.load(dataset_ids)
    selected_areas_tracks.sort(key=lambda x: x[3])
    print(selected_areas_tracks[-1])
    road_net_graph = ox.graph_from_bbox(InitParameters.NORTH, InitParameters.SOUTH, InitParameters.WEST,
                                        InitParameters.EAST,
                                        network_type='drive', simplify=True)
    roads_density_flow = list()
    map_con = build_matching_graph(road_net_graph)
    road_infos = grasp_road_infos(road_net_graph)

    sample_time = InitParameters.SAMPLE_TIME
    cars_set = [set() for _ in road_infos.keys()]
    for vehicle_time, vehicles in itertools.groupby(selected_areas_tracks, key=lambda x: x[3]):
        if int(vehicle_time) < sample_time:
            for vehicle in list(vehicles):
                matched_path, _ = map_path_to_graph(map_con, [(vehicle[1], vehicle[2])])
                # plot_matched_edges_on_graph(road_net_graph, matched_path)
                road_id = find_road_id(road_infos, matched_path)
                if road_id:
                    cars_set[road_id - 1].add(vehicle[0])

        cars_count = [len(car_set) for car_set in cars_set]
        road_zero_car_count = sum(1 for count in cars_count if count == 0)
        if road_zero_car_count > 6:
            if sample_time == int(vehicle_time):
                print(
                    f"=========The data belonging to the time {sample_time - InitParameters.TIME_INTERVAL} has been processed!=========")
                sample_time += InitParameters.TIME_INTERVAL
                for i in range(len(cars_set)):
                    cars_set[i].clear()
            continue
        if sample_time == int(vehicle_time):
            density_vehicle_time_road = len(InitParameters.SELECTED_ROAD_PATHS) * [0.0]
            density_flow_road = list()
            density_flow_road.append(sample_time)
            sample_time += InitParameters.TIME_INTERVAL
            for vehicle in list(vehicles):
                matched_path, _ = map_path_to_graph(map_con, [(vehicle[1], vehicle[2])])
                # plot_matched_edges_on_graph(road_net_graph, matched_path)
                road_id = find_road_id(road_infos, matched_path)
                if road_id:
                    density_value = 1000 / road_infos[road_id]['length']
                    density_vehicle_time_road[road_id - 1] += density_value

            for i in range(len(cars_set)):
                density_flow = (density_vehicle_time_road[i], len(cars_set[i]) * 6)
                density_flow_road.append(density_flow)
                cars_set[i].clear()
            printable_density_flow = ','.join(map(str, density_flow_road))
            print(printable_density_flow)
            roads_density_flow.append(density_flow_road)
            print(
                f"=========The data belonging to the time {sample_time - InitParameters.TIME_INTERVAL} has been processed!=========")

    return roads_density_flow


def store_processed_data(road_density_flow, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(road_density_flow, file)


def load_processed_data(file_path):
    with open(file_path, 'rb') as file:
        tracks_data = pickle.load(file)
        tracks_data = [tracks_data[i][1:] for i in range(len(tracks_data))]
        np_data = np.array(tracks_data)
        np_data = np.transpose(np_data, (1, 2, 0))
        np_data = np_data.astype(np.float32)

    return np_data


def visualize_selected_areas():
    road_net_graph = ox.graph_from_bbox(InitParameters.NORTH, InitParameters.SOUTH, InitParameters.WEST,
                                        InitParameters.EAST,
                                        network_type='drive', simplify=True)
    visualize_selected_paths(road_net_graph)
