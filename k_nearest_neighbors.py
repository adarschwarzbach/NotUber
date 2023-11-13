import math

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


def initalize_cluster_centers():
    # SELECT NODES
    min_lat = min(node['lat'] for node in node_data.values())
    max_lat = max(node['lat'] for node in node_data.values())
    min_lon = min(node['lon'] for node in node_data.values())
    max_lon = max(node['lon'] for node in node_data.values())
    # Example: 5x4 grid for 20 cells

    num_rows = 5
    num_cols = 4

    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols

    grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

    # Assign nodes to grid cells
    for node_id, node in node_data.items():
        row = int((node['lat'] - min_lat) / lat_step)
        col = int((node['lon'] - min_lon) / lon_step)

        # Adjust for edge cases
        row = min(row, num_rows - 1)
        col = min(col, num_cols - 1)

        grid[row][col].append(node_id)

    # SELECT NODES
    selected_nodes = []
    for row in grid:
        for cell in row:
            if cell:
                selected_nodes.append(cell[0])  # Selecting the first node in each cell

    # HANDLE SPARSE REGIONS





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

def update_cluster_centers(clusters, nodes):
    for cluster in clusters.values():
        members = cluster['members']
        avg_lat = sum(nodes[node_id]['lat'] for node_id in members) / len(members)
        avg_lon = sum(nodes[node_id]['lon'] for node_id in members) / len(members)
        cluster['center'] = {'lat': avg_lat, 'lon': avg_lon}

def geospatial_clustering(nodes, initial_cluster_centers, max_iterations=100):
    clusters = {i: {'center': center, 'members': set()} for i, center in enumerate(initial_cluster_centers)}

    for _ in range(max_iterations):
        changes = False

        # Assign nodes to the nearest cluster
        for node_id, node in nodes.items():
            assigned_cluster = assign_to_nearest_cluster(node, clusters)

            if node_id not in clusters[assigned_cluster]['members']:
                for cluster in clusters.values():
                    cluster['members'].discard(node_id)
                clusters[assigned_cluster]['members'].add(node_id)
                changes = True

        if not changes:
            break

        # Recalculate cluster centers
        update_cluster_centers(clusters, nodes)

    return clusters


