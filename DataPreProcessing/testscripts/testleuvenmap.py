import os
from pathlib import Path
import matplotlib.pyplot as plt
import requests
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
import osmread
from DataPreProcessing.testscripts.readdata import PNeumaDataset
from leuvenmapmatching import visualization as mmviz
# import osmnx as ox

def osm_get(xml_file, tracks):
    # 初始化最大和最小经纬度坐标
    min_lat = float('inf')
    max_lat = float('-inf')
    min_lon = float('inf')
    max_lon = float('-inf')

    # 遍历所有车辆的轨迹数据
    for vehicle_track in tracks:
        for lat, lon in vehicle_track:
            # 更新最大和最小纬度
            if lat < min_lat:
                min_lat = lat
            if lat > max_lat:
                max_lat = lat
            # 更新最大和最小经度
            if lon < min_lon:
                min_lon = lon
            if lon > max_lon:
                max_lon = lon


    url = f'http://overpass-api.de/api/map?bbox=23.728969,37.976506,23.736522,37.984008'

    # 发起请求
    r = requests.get(url, stream=True)

    # 保存文件
    with xml_file.open('wb') as ofile:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                ofile.write(chunk)

def osm_build(xml_file):
    map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)
    for entity in osmread.parse_file(str(xml_file)):
        if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
            for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
                map_con.add_edge(node_a, node_b)
                # Some roads are one-way. We'll add both directions.
                map_con.add_edge(node_b, node_a)
        if isinstance(entity, osmread.Node):
            map_con.add_node(entity.id, (entity.lat, entity.lon))
    map_con.purge()

    matcher = DistanceMatcher(map_con,
                              max_dist=100, max_dist_init=25,  # meter
                              min_prob_norm=0.001,
                              non_emitting_length_factor=0.75,
                              obs_noise=50, obs_noise_ne=75,  # meter
                              dist_noise=50,  # meter
                              non_emitting_states=True,
                              max_lattice_width=5)

    # states, lastidx = matcher.match(tracks[84])

    return map_con, matcher

def mapping_visualization(map_con, matcher, tracks):
    # matcher.match(tracks[0])
    # mmviz.plot_map(map_con, matcher=matcher,
    #                 use_osm=True, zoom_path=True,
    #                 show_labels=False, show_matching=True, show_graph=False,
    #                 filename="Track11.png")
    # 创建一个matplotlib图形和轴
    fig, ax = plt.subplots(figsize=(10, 10))

    for track in tracks[0:2]:
        # 遍历轨迹并匹配每一条
        matcher.match(track)
        # 绘制匹配的轨迹
        mmviz.plot_map(map_con, matcher=matcher, zoom_path=False, show_labels=False, use_osm=True, ax=ax, show_graph=False,
                       show_matching=False)

    # matcher.match(tracks[28])
    # mmviz.plot_map(map_con, matcher=matcher, zoom_path=False, show_labels=False, use_osm=True, ax=ax, show_graph=False,
    #                show_matching=False)
    # matcher.match(tracks[84])
    # mmviz.plot_map(map_con, matcher=matcher, zoom_path=False, show_labels=False, use_osm=True, ax=ax, show_graph=False,
    #                show_matching=False)

    # 设置图例和标题
    ax.set_title('Multiple Trajectory Matchings')
    # ax.legend(['Trajectory 1', 'Trajectory 2', 'Trajectory 3'])  # 根据轨迹数量调整
    plt.savefig('Trajectories1-10.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    file_path = os.path.join('../../Data', "20181024", "20181024_d2_0830_0900.csv")
    tracks = PNeumaDataset.read_pneuma_csv(file_path)
    # 定义文件保存位置和URL
    xml_file = Path("..") / "osm.xml"
    osm_get(xml_file, tracks)
    # if not xml_file.exists():
    #     osm_get(xml_file, tracks)
    # else:
    #     print("OpenStreetMap has been grasped!")
    map_con, matcher = osm_build(xml_file)
    # matcher = trace_mapping(map_con, tracks)
    mapping_visualization(map_con, matcher, tracks)


