import osmnx as ox
from leuvenmapmatching.map.inmem import InMemMap

# 定义地点和道路名称
place_name = "Leuven, Belgium"
road_name = "Bondgenotenlaan"  # 你想要导入的特定道路名称

# 加载地区的路网数据，不进行简化以保留所有节点和边
graph = ox.graph_from_place(place_name, network_type='drive', simplify=False)

# 过滤出包含特定道路名称的边
edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
specific_edges = edges[edges['name'] == road_name]

# 项目图到适当的坐标系统
graph_proj = ox.project_graph(graph)

# 获取项目后的所有节点和边
nodes_proj, edges_proj = ox.graph_to_gdfs(graph_proj, nodes=True, edges=True)

# 创建地图匹配对象

map_con = InMemMap("local_map", use_latlon=True, use_rtree=True, index_edges=True)

# 根据筛选的特定道路添加节点和边到地图对象
for eid, row in specific_edges.iterrows():
    # 确保每个节点被添加
    map_con.add_node(row['u'], (nodes_proj.loc[row['u']]['x'], nodes_proj.loc[row['u']]['y']))
    map_con.add_node(row['v'], (nodes_proj.loc[row['v']]['x'], nodes_proj.loc[row['v']]['y']))
    # 添加边
    map_con.add_edge((nodes_proj.loc[row['u']]['y'], nodes_proj.loc[row['u']]['x']),
                     (nodes_proj.loc[row['v']]['y'], nodes_proj.loc[row['v']]['x']))

# 注意：'x'和'y'对应于经度和纬度。这取决于你的地图匹配库设置是否要求坐标以特定顺序（经度，纬度或纬度，经度）。
