import random
import math
import json
from datetime import datetime



def initialize_cluster_centers(node_data):
    num_rows =  100 
    num_cols = 100

    min_lat = min(node['lat'] for node in node_data.values())
    max_lat = max(node['lat'] for node in node_data.values())
    min_lon = min(node['lon'] for node in node_data.values())
    max_lon = max(node['lon'] for node in node_data.values())
    print(min_lat, max_lat, min_lon, max_lon)
    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols

    grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]

    # Assign nodes to grid cells
    for node_id, node in node_data.items():
        row = min(int((node['lat'] - min_lat) / lat_step), num_rows - 1)
        col = min(int((node['lon'] - min_lon) / lon_step), num_cols - 1)
        grid[row][col].append(node_id)


    # determine density of grid
    new_grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

    for r in range(num_rows):
        for c in range(num_cols):
            new_grid[r][c] = len(grid[r][c])

    # print(new_grid)

    # determine ride density

    pickup_location_grid = [[0 for _ in range(num_cols)] for _ in range(num_rows)]
    with open('updated_passengers.json', 'r') as file:
        passenger_data = json.load(file)


    rides = 0
    for ride in passenger_data:
        row = min(int((float(ride['Source Lat']) - min_lat) / lat_step), num_rows - 1)
        col = min(int((float(ride['Source Lon']) - min_lon) / lon_step), num_cols - 1)
        pickup_location_grid[row][col] += 1

    average_cell_requests = 0
    all_requests = []

    for r in pickup_location_grid:
        for cell in r:
            average_cell_requests += cell
            all_requests.append(cell)


    average_cell_requests /= (num_cols * num_rows)
    all_requests.sort()
    median_cell_requests = all_requests[len(all_requests) - num_cols * 2]
    print(pickup_location_grid)
    print(average_cell_requests)
    print(median_cell_requests)

    with open('density_grid.json', 'w') as f:
        json.dump({ 'density_grid':pickup_location_grid, 'average_density':average_cell_requests, 'median_cell_requests':median_cell_requests}, f, indent=4)




    # Select nodes
    # selected_nodes = []
    # additional_nodes_needed = int(len(node_data) ** (1/1.352)) # ~550 nodes, exponentially less then original graph
    # for row in grid:
    #     for cell in row:
    #         if cell:
    #             selected_nodes.append(cell[0])  # Selecting the first node in each cell
    #             additional_nodes_needed -= 1

    # # If more nodes are needed, randomly select from populated cells
    # if additional_nodes_needed > 0:
    #     for row in grid:
    #         for cell in row:
    #             if len(cell) > 1:
    #                 random.shuffle(cell)  # Shuffle the nodes in the cell
    #                 for node_id in cell:
    #                     if additional_nodes_needed > 0 and node_id not in selected_nodes:
    #                         selected_nodes.append(node_id)
    #                         additional_nodes_needed -= 1
    #                     else:
    #                         break

    # return selected_nodes




def convert_to_hours(date_time_str):
    date_time_obj = datetime.strptime(date_time_str, '%m/%d/%Y %H:%M:%S')
    return date_time_obj.hour + date_time_obj.minute / 60 + date_time_obj.second / 3600


def time_based_density(node_data):
    num_rows =  25 
    num_cols = 25

    # Calculate the boundaries of the grid
    min_lat = min(node['lat'] for node in node_data.values())
    max_lat = max(node['lat'] for node in node_data.values())
    min_lon = min(node['lon'] for node in node_data.values())
    max_lon = max(node['lon'] for node in node_data.values())

    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols

    # Load passenger data
    with open('updated_passengers.json', 'r') as file:
        passenger_data = json.load(file)

    # Find the earliest and latest ride times
    earliest_date_time = min(datetime.strptime(ride['Date/Time'], '%m/%d/%Y %H:%M:%S') for ride in passenger_data)
    latest_date_time = max(datetime.strptime(ride['Date/Time'], '%m/%d/%Y %H:%M:%S') for ride in passenger_data)

    # Calculate total duration in hours
    total_duration = (latest_date_time - earliest_date_time).total_seconds() / 3600

    # Calculate the number of 4-hour blocks
    time_blocks = math.ceil(total_duration / 4)

    # Initialize grids for each time block
    time_block_grids = [[[0 for _ in range(num_cols)] for _ in range(num_rows)] for _ in range(time_blocks)]

    # Populate grids based on time blocks
    for ride in passenger_data:
        ride_time = datetime.strptime(ride['Date/Time'], '%m/%d/%Y %H:%M:%S')
        hours_since_start = (ride_time - earliest_date_time).total_seconds() / 3600
        block_index = int(hours_since_start / 4)
        
        row = min(int((float(ride['Source Lat']) - min_lat) / lat_step), num_rows - 1)
        col = min(int((float(ride['Source Lon']) - min_lon) / lon_step), num_cols - 1)

        time_block_grids[block_index][row][col] += 1

    # Calculate statistics for each grid
    density_stats = []

    for grid in time_block_grids:
        total_requests = sum(sum(cell for cell in row) for row in grid)
        all_requests = [cell for row in grid for cell in row]
        all_requests.sort()
        average_requests = total_requests / (num_rows * num_cols)
        median_requests = all_requests[len(all_requests) // 2] if all_requests else 0

        density_stats.append({
            'average_requests': average_requests,
            'median_requests': median_requests
        })

    # Write to JSON
    with open('time_density.json', 'w') as f:
        json.dump({
            'time_block_grids': time_block_grids,
            'density_stats': density_stats,
        }, f, indent=4)



def develop_method():
    print('Loading graph data...')
    with open('graph.json', 'r') as file:
        graph_data = json.load(file)

    node_data = {node_id: {'lat': graph_data[node_id]['coordinates']['lat'], 'lon': graph_data[node_id]['coordinates']['lon']} for node_id in graph_data}

    centers = time_based_density(node_data)


if __name__ == "__main__":
    develop_method()




"""

graph looks like:

[
[878, 589, 6, 0, 0, 0, 0, 0, 0, 0],
[325, 1184, 1191, 60, 194, 276, 212, 204, 0, 0],
[2, 857, 1451, 1048, 921, 1594, 294, 252, 609, 263],
[0, 409, 854, 400, 1218, 1521, 1297, 320, 421, 317],
[0, 0, 0, 0, 1069, 1493, 1692, 1901, 2039, 1053],
[0, 0, 0, 0, 1380, 1435, 1919, 1737, 1658, 1334],
[0, 0, 0, 0, 545, 1165, 1301, 1313, 1585, 736],
[0, 0, 0, 0, 6, 1282, 997, 816, 347, 0],
[0, 0, 0, 0, 0, 617, 1837, 1579, 149, 0],
[0, 0, 0, 0, 0, 0, 1022, 954, 0, 0]
]

"""


"""
Ride request density

[
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, 5, 1, 0, 0, 0, 0, 0, 1],
[0, 0, 1, 1, 0, 1, 8, 2, 2, 2, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 7, 17, 8, 4, 0, 2, 65, 1, 1],
[0, 8, 6, 0, 1, 0, 39, 159, 62, 8, 5, 2, 3, 1, 2],
[0, 1, 11, 0, 0, 3, 337, 168, 100, 10, 4, 0, 3, 0, 1],
[1, 0, 2, 0, 1, 24, 909, 983, 74, 4, 6, 4, 4, 1, 1],
[0, 0, 0, 0, 0, 2, 120, 1170, 104, 25, 53, 7, 0, 1, 0],
[0, 0, 2, 0, 1, 2, 1, 191, 124, 3, 0, 2, 0, 0, 0],
[1, 0, 1, 0, 3, 0, 0, 20, 36, 2, 1, 0, 0, 0, 2],
[0, 0, 1, 0, 0, 0, 0, 0, 8, 2, 0, 1, 0, 0, 1],
[0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 1, 2, 0, 0, 0],
[2, 0, 0, 0, 1, 2, 1, 1, 0, 1, 4, 1, 2, 1, 7]]

"""

"""
Destination Density

[
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 5, 3, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 0, 0, 7, 6, 4, 3, 0, 0, 0, 0, 5],
[1, 0, 0, 0, 0, 0, 11, 28, 16, 4, 0, 1, 124, 0, 4],
[0, 6, 5, 0, 2, 0, 20, 176, 52, 11, 1, 2, 4, 2, 0],
[0, 1, 18, 0, 2, 6, 304, 157, 126, 12, 1, 11, 2, 1, 1],
[0, 0, 1, 1, 0, 13, 694, 857, 98, 11, 9, 5, 3, 1, 3],
[0, 0, 0, 0, 0, 3, 114, 1275, 116, 22, 103, 5, 0, 2, 1],
[1, 0, 0, 0, 0, 1, 0, 227, 131, 9, 0, 2, 1, 0, 0],
[1, 0, 0, 1, 1, 0, 0, 15, 34, 1, 0, 0, 0, 0, 2],
[0, 0, 0, 0, 0, 1, 0, 0, 30, 5, 3, 2, 1, 0, 1],
[0, 0, 0, 0, 1, 0, 0, 1, 5, 4, 6, 0, 1, 0, 1],
[0, 0, 0, 0, 1, 3, 0, 1, 2, 7, 2, 5, 3, 0, 2]
]

"""