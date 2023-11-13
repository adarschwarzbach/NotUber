import math
import json
from collections import Counter
import random

def haversine(lon1, lat1, lon2, lat2):
    R = 3959.87433 # Radius of Earth in miles

    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dLon/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance


def initialize_cluster_centers(node_data):
    # Adjusted grid resolution
    num_rows = 35  # Adjust the number of rows as needed
    num_cols = 30  # Adjust the number of columns as needed

    min_lat = min(node['lat'] for node in node_data.values())
    max_lat = max(node['lat'] for node in node_data.values())
    min_lon = min(node['lon'] for node in node_data.values())
    max_lon = max(node['lon'] for node in node_data.values())

    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols

    grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

    # Assign nodes to grid cells
    for node_id, node in node_data.items():
        row = min(int((node['lat'] - min_lat) / lat_step), num_rows - 1)
        col = min(int((node['lon'] - min_lon) / lon_step), num_cols - 1)
        grid[row][col].append(node_id)

    # Select nodes
    selected_nodes = []
    additional_nodes_needed = 550
    for row in grid:
        for cell in row:
            if cell:
                selected_nodes.append(cell[0])  # Selecting the first node in each cell
                additional_nodes_needed -= 1

    # If more nodes are needed, randomly select from populated cells
    if additional_nodes_needed > 0:
        for row in grid:
            for cell in row:
                if len(cell) > 1:
                    random.shuffle(cell)  # Shuffle the nodes in the cell
                    for node_id in cell:
                        if additional_nodes_needed > 0 and node_id not in selected_nodes:
                            selected_nodes.append(node_id)
                            additional_nodes_needed -= 1
                        else:
                            break

    return selected_nodes




def assign_to_nearest_cluster(node, clusters):
    min_distance = float('inf')
    assigned_cluster = None

    for cluster_id, cluster in clusters.items():
        cluster_center = cluster['center']
        distance = haversine(node['lon'], node['lat'], cluster_center['lon'], cluster_center['lat'])

        if distance < min_distance:
            min_distance = distance
            assigned_cluster = cluster_id

    return assigned_cluster

def assign_nodes_to_clusters(nodes, initial_cluster_centers):
    # Initialize clusters
    clusters = {i: {'center': center, 'members': set()} for i, center in enumerate(initial_cluster_centers)}

    # Assign each node to the nearest cluster center
    for node_id, node in nodes.items():
        assigned_cluster = assign_to_nearest_cluster(node, clusters)
        clusters[assigned_cluster]['members'].add(node_id)

    return clusters


def test_all_nodes_assigned(nodes, clusters):
    unassigned_nodes = set(nodes.keys())
    
    for cluster in clusters.values():
        unassigned_nodes -= cluster['members']

    if not unassigned_nodes:
        print("All nodes have been successfully assigned to a cluster.")
    else:
        print("Some nodes are not assigned to any cluster:")
        for node_id in unassigned_nodes:
            print(f"Unassigned Node ID: {node_id}")



def average_cluster_connections(cluster_connections):
    averaged_connections = {}

    for cluster_id, connections in cluster_connections.items():
        averaged_connections[cluster_id] = {}
        for connected_cluster_id, connection_list in connections.items():
            connection_counter = Counter()
            for connection in connection_list:
                key = (connection['day_type'], connection['hour'])
                connection_counter[key] += 1
                if key not in averaged_connections[cluster_id]:
                    averaged_connections[cluster_id][key] = {k: v for k, v in connection.items() if k in ['length', 'max_speed', 'time']}
                else:
                    for k in ['length', 'max_speed', 'time']:
                        averaged_connections[cluster_id][key][k] += connection[k]

            # Calculate averages
            for key, count in connection_counter.items():
                for k in ['length', 'max_speed', 'time']:
                    averaged_connections[cluster_id][key][k] /= count

    return averaged_connections

def knn_wrapper():
    # Load node data from your JSON file
    print('Loading graph data...')
    with open('graph.json', 'r') as file:
        graph_data = json.load(file)
    

    # Assuming graph_data is a dictionary with node IDs as keys and {'lat': ..., 'lon': ...} as values
    node_data = {node_id: {'lat': graph_data[node_id]['coordinates']['lat'], 'lon': graph_data[node_id]['coordinates']['lon']} for node_id in graph_data}

    print('Initializing cluster centers...')
    selected_nodes = initialize_cluster_centers(node_data)

    print('Running geospatial clustering...')
    initial_cluster_centers = [{'lat': node_data[node_id]['lat'], 'lon': node_data[node_id]['lon']} for node_id in selected_nodes]
    clusters = clusters = assign_nodes_to_clusters(node_data, initial_cluster_centers)


    # luster_connections = aggregate_cluster_connections(nodes, clusters)
    # averaged_connections = average_cluster_connections(cluster_connections)

if __name__ == "__main__":
    knn_wrapper()






