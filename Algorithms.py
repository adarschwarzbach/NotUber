import csv
from datetime import datetime
import json
from math import radians, cos, sin, asin, sqrt, log
import heapq
from datetime import datetime
import random
import multiprocessing
import concurrent.futures
import time



# Function to read a CSV file
def read_csv_file(filepath):
    data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


# Read in un-proccessed data
def read_base_data():
    drivers_filepath = 'drivers.csv'
    passengers_filepath = 'passengers.csv'

    drivers_data = read_csv_file(drivers_filepath)
    passengers_data = read_csv_file(passengers_filepath)

    return drivers_data, passengers_data


# Helper to read a JSON file
def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data


# Build the graph from the node and connection data
def build_graph():
    print('Building graph...')
    # Reading the node data and connection data from JSON files
    node_coordinates_filepath = 'node_data.json'
    node_connections_filepath = 'adjacency.json'

    node_coordinates = read_json_file(node_coordinates_filepath)
    node_connections = read_json_file(node_connections_filepath)

    graph = {}

    # Add the nodes to the graph
    for node_id, coords in node_coordinates.items():
        graph[node_id] = {
            'coordinates': coords,
            'connections': {}
        }


    # Add the edges to the graph
    for start_node_id, connections in node_connections.items():
        for end_node_id, attributes in connections.items():
             graph[start_node_id]['connections'][end_node_id] = attributes
            #  graph[end_node_id]['connections'][start_node_id] = attributes # not necesary as there are seperate times for different directions

    print('Writing graph data to file...')
    graph_file = 'graph.json'
    with open(graph_file, 'w') as f:
        json.dump(graph, f, indent=4)  # Use indent=4 for pretty-printing


# Helper to calculate the great-circle distance between two points
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956 # in miles
    return c * r


# Helper to find the nearest node in the graph to a given lat/lon
def find_nearest_node(graph, latitude, longitude):
    nearest_node = None
    nearest_distance = float('inf')
    
    for node_id, node_info in graph.items():
        node_lat = node_info['coordinates']['lat']
        node_lon = node_info['coordinates']['lon']
        distance = haversine(longitude, latitude, node_lon, node_lat)
        if distance < nearest_distance:
            nearest_node = node_id
            nearest_distance = distance
    
    return nearest_node


# Pre-process data to determine closest node to driver/pickup/destination
def simple_pre_processing(graph, drivers_data, passengers_data):
    print('Finding nearest node for each driver...')
    # for each node in drivers_data, find the nearest node in graph and add it to the dictionary
    for driver in drivers_data:
        driver['node'] = find_nearest_node(graph, float(driver['Source Lat']), float(driver['Source Lon']))


    print('Finding nearest node for each passenger...')
    # for each node in passengers_data, find the nearest node in graph and add it to the dictionary
    for passenger in passengers_data:
        passenger['node'] = find_nearest_node(graph, float(passenger['Source Lat']), float(passenger['Source Lon']))
        passenger['destination_node'] = find_nearest_node(graph, float(passenger['Dest Lat']), float(passenger['Dest Lon']))


    drivers_file = 'updated_drivers.json'
    passengers_file = 'updated_passengers.json'


    print('Writing updated driver data to file...')
    with open(drivers_file, 'w') as f:
        json.dump(drivers_data, f, indent=4)  # Use indent=4 for pretty-printing

    print('Writing updated passenger data to file...')
    with open(passengers_file, 'w') as f:
        json.dump(passengers_data, f, indent=4)

# Determine closest node in tine graph to each driver/pickup/destination
def tiny_graph_pre_processing():
    print('Finding nearest cluster for each driver...')
    drivers_data, passengers_data = load_updated_data()
    # load tiny_graph.json
    with open('tiny_graph.json', 'r') as f:
        graph = json.load(f)

    # load updated_drivers.json and updated_passengers.json
    with open('updated_drivers.json', 'r') as f:
        drivers_data = json.load(f)
    
    with open('updated_passengers.json', 'r') as f:
        passengers_data = json.load(f)

    for driver in drivers_data:
        for node in graph.keys():
            if driver['node'] in graph[node]['members']:
                driver['cluster'] = node
                break

    for passenger in passengers_data:
        for node in graph.keys():
            if passenger['node'] in graph[node]['members']:
                passenger['cluster'] = node
                break
    
    for passenger in passengers_data:
        for node in graph.keys():
            if passenger['destination_node'] in graph[node]['members']:
                passenger['destination_cluster'] = node
                break

    
    # write the tiny_graph_drivers and tiny_graph_passengers to file
    with open('tiny_graph_drivers.json', 'w') as f:
        json.dump(drivers_data, f, indent=4)

    with open('tiny_graph_passengers.json', 'w') as f:
        json.dump(passengers_data, f, indent=4)


# Load pre-processed data
def load_updated_data():
    drivers_file = 'updated_drivers.json'
    passengers_file = 'updated_passengers.json'

    with open(drivers_file, 'r') as f:
        drivers_data = json.load(f)

    with open(passengers_file, 'r') as f:
        passengers_data = json.load(f)

    return drivers_data, passengers_data

# Load kd tree pre-processed data
def load_kd_updated_data():
    drivers_file = 'kd_drivers.json'
    passengers_file = 'kd_passengers.json'

    with open(drivers_file, 'r') as f:
        drivers_data = json.load(f)

    with open(passengers_file, 'r') as f:
        passengers_data = json.load(f)

    return drivers_data, passengers_data


# Load tiny graph pre-processed data
def load_tiny_graph_updated_data():
    drivers_file = 'tiny_graph_drivers.json'
    passengers_file = 'tiny_graph_passengers.json'

    with open(drivers_file, 'r') as f:
        drivers_data = json.load(f)

    with open(passengers_file, 'r') as f:
        passengers_data = json.load(f)

    return drivers_data, passengers_data


def load_graph():
    graph_file = 'graph.json'
    with open(graph_file, 'r') as f:
        graph = json.load(f)

    return graph

# Helper to convert a datetime string to a Unix timestamp
def parse_datetime_to_unix(datetime_str):
    try:
        dt = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
        day_type = 'weekday' if dt.weekday() < 5 else 'weekend'
        hour = dt.hour
        unix_timestamp = int(dt.timestamp())
        return unix_timestamp, hour, day_type
    except ValueError as e:
        print(f"Error parsing datetime: {e}")
        return None, None, None


def construct_queues(drivers_data, passengers_data):
    # Convert Date/Time for all passengers and drivers to Unix timestamps within the data
    for p in passengers_data:
        unix_time, hour, day_type = parse_datetime_to_unix(p['Date/Time'])
        if unix_time is not None:
            p['Date/Time'] = unix_time
            p['Hour'] = hour
            p['DayType'] = day_type

    for d in drivers_data:
        unix_time, hour, day_type = parse_datetime_to_unix(d['Date/Time'])
        if unix_time is not None:
            d['Date/Time'] = unix_time
            d['Hour'] = hour
            d['DayType'] = day_type
            d['number_of_trips'] = 0

    # Construct queues
    # Use index as a secondary sort key to ensure dictionaries are not compared
    passenger_queue = [(p['Date/Time'], i, p) for i, p in enumerate(passengers_data) if 'Date/Time' in p]
    passenger_queue = passenger_queue[::-1]  # Reverse the list so that the earliest passengers are at the front

    driver_queue = [(d['Date/Time'], i, d) for i, d in enumerate(drivers_data) if 'Date/Time' in d]
    # print(driver_queue[0], passenger_queue[0])
    heapq.heapify(driver_queue)


    return passenger_queue, driver_queue


# Djikstra's implementation to determine time for a given trip
def dijkstra(graph, start, end, hour, day_type):
    queue = [(0, start)]  # (cumulative_time, node)
    visited = set()
    distances = {start: 0}  # Initialize the start node

    while queue:
        # Get the node with the smallest cumulative time
        cumulative_time, node = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)

            # Reached destination
            if node == end:
                return cumulative_time

            for neighbor, edges in graph[node]['connections'].items():
                # Filter edges based on day_type and hour
                valid_edges = [edge for edge in edges if edge['day_type'] == day_type and edge['hour'] == hour]
                if not valid_edges:
                    continue

                edge = valid_edges[0]
                travel_time = edge['time'] * 60  # Convert hours to minutes
                new_time = cumulative_time + travel_time

                # Initialize and update the distance to the neighbor
                if neighbor not in distances or new_time < distances[neighbor]:
                    distances[neighbor] = new_time
                    heapq.heappush(queue, (new_time, neighbor))

    return float('inf')


# Helper to determine how many stops there are on the optimal path
def dijkstra_for_num_stops(graph, start, end, hour, day_type):
    # Initialize the priority queue with the start node, zero time, and zero stops
    queue = [(0, start, 0)]  # (cumulative_time, node, num_stops)
    visited = set()
    # Map to store shortest distance and number of stops to a node
    distances = {node: (float('inf'), float('inf')) for node in graph}
    distances[start] = (0, 0)

    while queue:
        # Get the node with the smallest cumulative time
        cumulative_time, node, num_stops = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)

            # Reached destination
            if node == end:
                return cumulative_time, num_stops

            for neighbor, edges in graph[node]['connections'].items():
                # Filter edges based on day_type and hour
                valid_edges = [edge for edge in edges if edge['day_type'] == day_type and edge['hour'] == hour]

                if not valid_edges:
                    continue

                edge = valid_edges[0]
                travel_time = edge['time'] * 60  # Convert hours to minutes
                new_time = cumulative_time + travel_time
                new_stops = num_stops + 1

                if new_time < distances[neighbor][0]:
                    distances[neighbor] = (new_time, new_stops)
                    heapq.heappush(queue, (new_time, neighbor, new_stops))

    # If the destination is not reachable, return infinity for both time and stops
    return float('inf'), float('inf')


# Djikstra's implementation for the tiny graph
def tiny_graph_dijkstra(graph, start, end, hour, day_type):
    queue = [(0, start, 0)]  # (cumulative_time, node, num_stops)
    visited = set()
    # Map to store shortest distance to a node along with number of stops
    distances = {node: (float('inf'), float('inf')) for node in graph}
    distances[start] = (0, 0)

    while queue:
        # Get the node with the smallest cumulative time
        cumulative_time, node, num_stops = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)

            # Reached destination
            if node == end:
                return cumulative_time, num_stops

            for neighbor in graph[node]['connections']:
                # Select edge based on day_type and hour
                edge = graph[node]['connections'][neighbor][f"{day_type}_{hour}"]
                travel_time = edge * 60 * 6.42   # Convert hours to minutes and multiply by 6.42 as there are on average 6.42 fewer edges in a given trip
                new_time = cumulative_time + travel_time
                new_stops = num_stops + 1

                if new_time < distances[neighbor][0]:
                    distances[neighbor] = (new_time, new_stops)
                    heapq.heappush(queue, (new_time, neighbor, new_stops))

    # If the destination is not reachable, return infinity for both time and stops
    return float('inf'), float('inf')


def calculate_driver_distances(driver, passenger_node, graph):
    driver_time, id, d = driver

    # Calculate the time it would take for the driver to reach the passenger
    projected_travel_to_pickup_time = dijkstra(graph, d['node'], passenger_node, d['Hour'], d['DayType'])
    
    return (projected_travel_to_pickup_time, driver_time, id, d)

class Node:
  # Build the tree node class
    def __init__(self, node_id, point, left=None, right=None):
        self.node_id = node_id
        self.point = point
        self.left = left
        self.right = right


def build_kd_tree(graph, depth=0, node_ids=None):
    if node_ids is None:
        node_ids = list(graph.keys())

    if not node_ids:
        return None

    k = 2  # Alternate each level of the tree because we are in 2D space
    axis = depth % k

    # Sort node_ids by the current axis
    sorted_node_ids = sorted(node_ids, key=lambda node_id: graph[node_id]['coordinates'][axis])
    median_idx = len(sorted_node_ids) // 2
    median_node_id = sorted_node_ids[median_idx]

    # Create a new node and construct subtrees
    return Node(
        node_id=median_node_id,
        point=graph[median_node_id]['coordinates'],
        left=build_kd_tree(graph, depth + 1, sorted_node_ids[:median_idx]),
        right=build_kd_tree(graph, depth + 1, sorted_node_ids[median_idx + 1:])
    )



def kd_closest_node(kd_tree, query_point, depth=0, best=None):
    if kd_tree is None:
        return best

    k = 2  # 2D space
    axis = depth % k

    # Check if current node is closer
    current_distance = haversine(query_point[0], query_point[1], kd_tree.point[0], kd_tree.point[1])
    if best is None or current_distance < best[1]:
        best = (kd_tree.node_id, current_distance)

    # Determine which subtree to search first
    next_branch = None
    other_branch = None
    if query_point[axis] < kd_tree.point[axis]:
        next_branch = kd_tree.left
        other_branch = kd_tree.right
    else:
        next_branch = kd_tree.right
        other_branch = kd_tree.left

    # Search next branch
    best = kd_closest_node(next_branch, query_point, depth + 1, best)

    # Check if other branch could have closer node
    if other_branch is not None:
        # Calculate distance to the plane (finds perpendicular distance to the plane in order to possible check other branch of tree)
        plane_distance = abs(query_point[axis] - kd_tree.point[axis])
        if plane_distance < best[1]:
            best = kd_closest_node(other_branch, query_point, depth + 1, best)

    return best


def kd_tree_graph_builder(graph):
    for node in graph.keys():
        graph[node]['coordinates'] = (graph[node]['coordinates']['lat'], graph[node]['coordinates']['lon'])

    return graph


# Pre-process data to determine closest node to driver/pickup/destination
def kd_tree_pre_processing(graph, drivers_data, passengers_data):
    kd_tree_graph = kd_tree_graph_builder(graph)
    kd_tree = build_kd_tree(kd_tree_graph)

    print('Finding nearest node for each driver with kd tree...')
    # for each node in drivers_data, find the nearest node in graph and add it to the dictionary
    for driver in drivers_data:
        query_point = (float(driver['Source Lat']), float(driver['Source Lon']))
        driver['node'] = kd_closest_node(kd_tree, query_point)[0]


    print('Finding nearest node for each passenger with kd tree...')
    # for each node in passengers_data, find the nearest node in graph and add it to the dictionary
    for passenger in passengers_data:
        query_point = (float(passenger['Source Lat']), float(passenger['Source Lon']))
        passenger['node'] = kd_closest_node(kd_tree, query_point)[0]
        query_point = (float(passenger['Dest Lat']), float(passenger['Dest Lon']))
        passenger['destination_node'] = kd_closest_node(kd_tree, query_point)[0]


    drivers_file = 'kd_drivers.json'
    passengers_file = 'kd_passengers.json'


    print('Writing updated driver data to file...')
    with open(drivers_file, 'w') as f:
        json.dump(drivers_data, f, indent=4)  
    print('Writing updated passenger data to file...')
    with open(passengers_file, 'w') as f:
        json.dump(passengers_data, f, indent=4)


# determine given time block
def get_time_block_index(current_epoch_time, earliest_epoch_time, block_duration_hours=4):
    # Convert epoch times to datetime objects
    current_time = datetime.utcfromtimestamp(current_epoch_time)
    earliest_time = datetime.utcfromtimestamp(earliest_epoch_time)

    # Calculate the number of hours since the earliest time
    hours_since_earliest = (current_time - earliest_time).total_seconds() / 3600

    # Determine the time block
    time_block_index = int(hours_since_earliest / block_duration_hours)
    return time_block_index


def calculate_trip_ratio(passenger, driver_node, driver, graph):
    # Compute the time from driver to passenger (pickup time)
    travel_to_pickup = dijkstra(graph, driver_node, passenger['node'], driver['Hour'], driver['DayType'])

    # Compute the trip time from passenger to destination (dropoff time)
    dropoff_time = dijkstra(graph, passenger['node'], passenger['destination_node'], driver['Hour'], driver['DayType'])

    # Calculate the ratio of trip time to pickup time
    if travel_to_pickup > 0:
        ratio = dropoff_time / travel_to_pickup 
    else:
        ratio = float('inf')  # Handle the case where pickup time is 0 to avoid division by zero

    return ratio, travel_to_pickup, dropoff_time, passenger


# generate a penalty for travelling somewhere with a low density of rides
def optimal_with_density_penalty(passengers, curr_time):
    with open('time_density.json', 'r') as file:
        d_grid = json.load(file)



    density_grids = d_grid['time_block_grids']
    average_ds = d_grid['density_stats']

    earliest_time = 1398409200

    time_index = get_time_block_index(curr_time, earliest_time)

    average_density = average_ds[time_index]['average_requests']
    density_grid = density_grids[time_index]
    
    num_rows = len(density_grid)
    num_cols = len(density_grid[0])

    # stay the same as we have the same graph
    min_lat = 40.4983687
    max_lat = 40.912507
    min_lon = -74.2552929
    max_lon = -73.7004728

    lat_step = (max_lat - min_lat) / num_rows
    lon_step = (max_lon - min_lon) / num_cols

    updated_passengers = []
    for current_ratio, travel_to_pickup_time, dropoff_time, passenger in passengers:
        row = min(int((float(passenger['Dest Lat']) - min_lat) / lat_step), num_rows - 1)
        col = min(int((float(passenger['Dest Lon']) - min_lon) / lon_step), num_cols - 1)

        # weighted_ratio = log(density_meaning) / 4 + current_ratio #  ** 2

        weighted_ratio = log(max(density_grid[row][col] / average_density, .5)) / 4 + current_ratio

        updated_passengers.append((weighted_ratio,travel_to_pickup_time, dropoff_time, passenger))
        # print( 'weighted ratio:',weighted_ratio, 'OG ratio:', current_ratio, 'Density:', density_grid[row][col])
    
    
    updated_passengers.sort(key=lambda x: x[0], reverse=True)

    # print(updated_passengers[0][0], updated_passengers[-1][0])
    return updated_passengers[0] 


# Baseline simulation
def simulate_t1(graph, passenger_queue, driver_queue):
    print('Running T1 simulation...')
    matches = []  # Track every trip
    total_time_drivers_travel_to_passengers = 0
    total_in_car_time = 0
    failute_count = 0
    exited_drivers = []

    total_stops = 0

    #test on smaller queue
    passenger_queue = passenger_queue[-400:]
    
    while passenger_queue:  # Continue until one of the queues is empty
        # Passenger and driver details
        passenger_request_time, _, passenger = passenger_queue.pop()  
        driver_time, _, driver = heapq.heappop(driver_queue)  # Pop the first available driver
        
        # Get the driver's current location and passenger's pickup location
        driver_location = driver['node']
        passenger_pickup = passenger['node']
        
        # Calculate time from passenger making request to driver becoming available
        wait_from_passenger_request = 0
        if passenger_request_time < driver_time:
            wait_from_passenger_request = (driver_time - passenger_request_time) / 60

            # print('wait_from_passenger_request', wait_from_passenger_request)


        # Calculate time for driver to reach passenger
        travel_to_pickup_time, stops_a = dijkstra_for_num_stops(graph, driver_location, passenger_pickup, driver['Hour'], driver['DayType'])
        
        if travel_to_pickup_time == float('inf'):
            print('No path to passenger', passenger, driver)
            failute_count += 1
            continue

        
        # Calculate time for driver to drop passenger at the destination
        passenger_destination = passenger['destination_node']
        dropoff_time, stops_b = dijkstra_for_num_stops(graph, passenger_pickup, passenger_destination, passenger['Hour'], passenger['DayType'])

        if dropoff_time == float('inf'):
            print('No path to destination', passenger, driver)
            failute_count += 1
            continue
        
        # Calculate the driver's new available time
        new_driver_time = driver_time + travel_to_pickup_time  * 60 + dropoff_time * 60
        
        # Update the driver's information
        driver['node'] = passenger_destination
        driver['Hour'] = passenger['Hour']
        driver['DayType'] = passenger['DayType']
        driver['Date/Time'] = new_driver_time
        driver['number_of_trips'] += 1
        
        # Add this trip to the matches list
        matches.append({
            'driver_location': driver_location,
            'passenger_pickup': passenger_pickup,
            'passenger_destination': passenger_destination,
            'pickup_wait_time': travel_to_pickup_time,
            'dropoff_time': dropoff_time,
            'wait_from_passenger_request': wait_from_passenger_request,
            'total_wait': travel_to_pickup_time + dropoff_time + wait_from_passenger_request,
        })
        
        # simulate drivers stopping to drive
        if driver['number_of_trips'] > 10 and len(driver_queue) > 20:
            random_number = random.randint(1, 10)

            # 1/10 chance of driver stopping after 10 trips
            if random_number == 1:
                heapq.heappush(driver_queue, (new_driver_time, id(driver), driver))
            else:
                exited_drivers.append(driver)
        else:
            # Re-insert the driver into the priority queue with the new available time
            heapq.heappush(driver_queue, (new_driver_time, id(driver), driver))
        
        total_time_drivers_travel_to_passengers += travel_to_pickup_time
        total_in_car_time += dropoff_time
        
        total_stops += stops_a + stops_b

        if len(passenger_queue) % 100 == 0:
            print(len(passenger_queue), 'passengers in queue')
            print(len(driver_queue), 'drivers in queue')
            print(stops_a, stops_b)


    trips_per_driver = []
    all_drivers = [driver for _, _, driver in driver_queue] + exited_drivers
    for driver in all_drivers:
        trips_per_driver.append(driver['number_of_trips'])
        

    print('average stops', total_stops / 400 )
    return matches


# When there is a choice, passenger will be assigned to closest availible driver
def simulate_t2(graph, passenger_queue, driver_queue):
    print('Running T2 simulation...')
    matches = []  # Track every trip
    total_time_drivers_travel_to_passengers = 0
    total_in_car_time = 0
    failute_count = 0
    exited_drivers = []

    # passenger_queue = passenger_queue[4900:5100]

    
    while passenger_queue:  # Continue until one of the queues is empty
        # Passenger and driver details
        passenger_request_time, _, passenger = passenger_queue.pop()  

        available_drivers = []

        while driver_queue and driver_queue[0][0] <= passenger_request_time:
            available_drivers.append(heapq.heappop(driver_queue))

        # Pop all available drivers whose availability time is less than or equal to the passenger request time
        driver = None
        if available_drivers:
            # print(len(available_drivers), 'drivers available')
            # Calculate Haversine distance from each available driver to the passenger
            passenger_coords = graph[passenger['node']]['coordinates']

            driver_distances = []
            for driver_time, id, driver in available_drivers:
                driver_coords = graph[driver['node']]['coordinates']

                distance = haversine(passenger_coords['lat'], passenger_coords['lon'], 
                                     driver_coords['lat'], driver_coords['lon'])
                driver_distances.append((distance, driver_time, id, driver))


            driver_distances.sort()
            # Select the driver with the shortest distance
            _, driver_time, id, driver = driver_distances[0]
            
            # Re-insert the other drivers into the priority queue
            for d_time, id, d in available_drivers:
                if d != driver:
                    heapq.heappush(driver_queue, (d_time, id, d))

        else:
            # If no drivers are available yet, take the earliest available driver
            driver_time, id, driver = heapq.heappop(driver_queue)

        # Get the driver's current location and passenger's pickup location
        driver_location = driver['node']
        passenger_pickup = passenger['node']
        
        # Calculate time from passenger making request to driver becoming available
        wait_from_passenger_request = 0
        if passenger_request_time < driver_time:
            wait_from_passenger_request = (driver_time - passenger_request_time) / 60

            # print('wait_from_passenger_request', wait_from_passenger_request)


        # Calculate time for driver to reach passenger
        travel_to_pickup_time = dijkstra(graph, driver_location, passenger_pickup, driver['Hour'], driver['DayType'])
        
        if travel_to_pickup_time == float('inf'):
            print('No path to passenger', passenger, driver)
            failute_count += 1
            continue

        
        # Calculate time for driver to drop passenger at the destination
        passenger_destination = passenger['destination_node']
        dropoff_time = dijkstra(graph, passenger_pickup, passenger_destination, passenger['Hour'], passenger['DayType'])

        if dropoff_time == float('inf'):
            print('No path to destination', passenger, driver)
            failute_count += 1
            continue
        
        # Calculate the driver's new available time
        new_driver_time = driver_time + travel_to_pickup_time  * 60 + dropoff_time * 60
        
        # Update the driver's information
        driver['node'] = passenger_destination
        driver['Hour'] = passenger['Hour']
        driver['DayType'] = passenger['DayType']
        driver['Date/Time'] = new_driver_time
        driver['number_of_trips'] += 1
        
        # Add this trip to the matches list
        matches.append({
            'driver_location': driver_location,
            'passenger_pickup': passenger_pickup,
            'passenger_destination': passenger_destination,
            'pickup_wait_time': travel_to_pickup_time,
            'dropoff_time': dropoff_time,
            'wait_from_passenger_request': wait_from_passenger_request,
            'total_wait': travel_to_pickup_time + dropoff_time + wait_from_passenger_request,
        })
        
        # simulate drivers stopping to drive
        if driver['number_of_trips'] > 10 and len(driver_queue) > 20:
            random_number = random.randint(1, 10)

            # 1/10 chance of driver stopping after 10 trips
            if random_number == 1:
                heapq.heappush(driver_queue, (new_driver_time, id, driver))
            else:
                exited_drivers.append(driver)
        else:
            # Re-insert the driver into the priority queue with the new available time
            heapq.heappush(driver_queue, (new_driver_time, id, driver))
        
        total_time_drivers_travel_to_passengers += travel_to_pickup_time
        total_in_car_time += dropoff_time
        

        if len(passenger_queue) % 100 == 0:
            print(len(passenger_queue), 'passengers in queue')
            print(len(driver_queue), 'drivers in queue')


    trips_per_driver = []
    all_drivers = [driver for _, _, driver in driver_queue] + exited_drivers
    for driver in all_drivers:
        trips_per_driver.append(driver['number_of_trips'])
        

    return matches


# When there is a choice, passenger will be assigned to the driver with the shortest time to drive to them
def simulate_t3(graph, passenger_queue, driver_queue):
    print('Running T3 simulation...')
    matches = []  # Track every trip
    total_time_drivers_travel_to_passengers = 0
    total_in_car_time = 0
    failute_count = 0
    exited_drivers = []

    passenger_queue = passenger_queue[100:]

    
    while passenger_queue:  # Continue until one of the queues is empty
        # Passenger and driver details
        passenger_request_time, _, passenger = passenger_queue.pop()  


        available_drivers = []

        while driver_queue and driver_queue[0][0] <= passenger_request_time:
            available_drivers.append(heapq.heappop(driver_queue))

        # Pop all available drivers whose availability time is less than or equal to the passenger request time
        driver = None
        given_travel_to_pickup_time = None
        if available_drivers:
            # print(len(available_drivers), 'drivers available')
            driver_distances = []
            for driver_time, id, d in available_drivers:

                # Calculate the time it would take for the driver to reach the passenger
                projected_travel_to_pickup_time = dijkstra(graph, d['node'], passenger['node'], d['Hour'], d['DayType'])
                
                driver_distances.append((projected_travel_to_pickup_time, driver_time, id, d))


            driver_distances.sort()
            # Select the driver with the shortest time to get there
            shortest_travel_to_pickup_time, driver_time, id, driver = driver_distances[0]
            given_travel_to_pickup_time = shortest_travel_to_pickup_time
            
            # Re-insert the other drivers into the priority queue
            for d_time, id, d in available_drivers:
                if d != driver:
                    heapq.heappush(driver_queue, (d_time, id, d))

        else:
            # If no drivers are available yet, take the earliest available driver
            driver_time, id, driver = heapq.heappop(driver_queue)

        ###
        # Get the driver's current location and passenger's pickup location
        driver_location = driver['node']
        passenger_pickup = passenger['node']
        
        # Calculate time from passenger making request to driver becoming available
        wait_from_passenger_request = 0
        if passenger_request_time < driver_time:
            wait_from_passenger_request = (driver_time - passenger_request_time) / 60



        # Calculate time for driver to reach passenger, unless we have already computed it
        if given_travel_to_pickup_time is None:
            travel_to_pickup_time = dijkstra(graph, driver_location, passenger_pickup, driver['Hour'], driver['DayType'])
        else:
            travel_to_pickup_time = given_travel_to_pickup_time
        
        if travel_to_pickup_time == float('inf'):
            print('No path to passenger', passenger, driver)
            failute_count += 1
            continue

        
        # Calculate time for driver to drop passenger at the destination
        passenger_destination = passenger['destination_node']
        dropoff_time = dijkstra(graph, passenger_pickup, passenger_destination, passenger['Hour'], passenger['DayType'])

        if dropoff_time == float('inf'):
            print('No path to destination', passenger, driver)
            failute_count += 1
            continue
        
        # Calculate the driver's new available time
        new_driver_time = driver_time + travel_to_pickup_time  * 60 + dropoff_time * 60
        
        # Update the driver's information
        driver['node'] = passenger_destination
        driver['Hour'] = passenger['Hour']
        driver['DayType'] = passenger['DayType']
        driver['Date/Time'] = new_driver_time
        driver['number_of_trips'] += 1
        
        # Add this trip to the matches list
        matches.append({
            'driver_location': driver_location,
            'passenger_pickup': passenger_pickup,
            'passenger_destination': passenger_destination,
            'pickup_wait_time': travel_to_pickup_time,
            'dropoff_time': dropoff_time,
            'wait_from_passenger_request': wait_from_passenger_request,
            'total_wait': travel_to_pickup_time + dropoff_time + wait_from_passenger_request,
        })
        
        # simulate drivers stopping to drive
        if driver['number_of_trips'] > 10 and len(driver_queue) > 20:
            random_number = random.randint(1, 10)

            # 1/10 chance of driver stopping after 10 trips
            if random_number == 1:
                heapq.heappush(driver_queue, (new_driver_time, id, driver))
            else:
                exited_drivers.append(driver)
        else:
            # Re-insert the driver into the priority queue with the new available time
            heapq.heappush(driver_queue, (new_driver_time, id, driver))
        
        total_time_drivers_travel_to_passengers += travel_to_pickup_time
        total_in_car_time += dropoff_time
        

        if len(passenger_queue) % 100 == 0:
            print(len(passenger_queue), 'passengers in queue')
            print(len(driver_queue), 'drivers in queue')


    trips_per_driver = []
    all_drivers = [driver for _, _, driver in driver_queue] + exited_drivers
    for driver in all_drivers:
        trips_per_driver.append(driver['number_of_trips'])
        

    return matches


# Optimize: use approximate times with the tiny graph of clusters created with KD tree pre-process
def simulate_t4(passenger_queue, driver_queue):
    print('Running T4 B simulation...')

    with open('graph.json', 'r') as f:
        g = json.load(f)
    # load tiny_graph.json
    with open('tiny_graph_2.json', 'r') as f:
        tiny_graph = json.load(f)

    matches = []  # Track every trip
    total_time_drivers_travel_to_passengers = 0
    total_in_car_time = 0
    failute_count = 0
    exited_drivers = []
    total_stops = 0

    # passenger_queue = passenger_queue[4900:5100]

    
    while passenger_queue:  # Continue until one of the queues is empty
        # Passenger and driver details
        passenger_request_time, _, passenger = passenger_queue.pop()  


        available_drivers = []

        while driver_queue and driver_queue[0][0] <= passenger_request_time:
            available_drivers.append(heapq.heappop(driver_queue))

        # Pop all available drivers whose availability time is less than or equal to the passenger request time
        driver = None
        given_travel_to_pickup_time = None
        if available_drivers:
            # print(len(available_drivers), 'drivers available')
            driver_distances = []
            for driver_time, id, d in available_drivers:

                # Calculate the time it would take for the driver to reach the passenger
                projected_travel_to_pickup_time, _ = tiny_graph_dijkstra(tiny_graph, d['cluster'], passenger['cluster'], d['Hour'], d['DayType'])
                
                driver_distances.append((projected_travel_to_pickup_time, driver_time, id, d))


            driver_distances.sort()
            # Select the driver with the shortest time to get there
            shortest_travel_to_pickup_time, driver_time, id, driver = driver_distances[0]
            given_travel_to_pickup_time = shortest_travel_to_pickup_time
            
            # Re-insert the other drivers into the priority queue
            for d_time, id, d in available_drivers:
                if d != driver:
                    heapq.heappush(driver_queue, (d_time, id, d))

        else:
            # If no drivers are available yet, take the earliest available driver
            driver_time, id, driver = heapq.heappop(driver_queue)

        ###
        # Get the driver's current location and passenger's pickup location
        driver_location = driver['cluster']
        passenger_pickup = passenger['cluster']
        
        # Calculate time from passenger making request to driver becoming available
        wait_from_passenger_request = 0
        if passenger_request_time < driver_time:
            wait_from_passenger_request = (driver_time - passenger_request_time) / 60



        # Calculate time for driver to reach passenger, unless we have already computed it
        if given_travel_to_pickup_time is None:
            travel_to_pickup_time, _ = tiny_graph_dijkstra(tiny_graph, driver_location, passenger_pickup, driver['Hour'], driver['DayType'])
        else:
            travel_to_pickup_time = given_travel_to_pickup_time
        
        if travel_to_pickup_time == float('inf'):
            print('No path to passenger', passenger, driver)
            failute_count += 1
            continue

        
        # Calculate time for driver to drop passenger at the destination
        passenger_destination = passenger['destination_cluster']
        dropoff_time, stops_a = tiny_graph_dijkstra(tiny_graph, passenger_pickup, passenger_destination, passenger['Hour'], passenger['DayType'])

        total_stops += stops_a
        if dropoff_time == float('inf'):
            print('No path to destination', passenger, driver)
            failute_count += 1
            continue

        
        # Calculate the driver's new available time
        new_driver_time = driver_time + travel_to_pickup_time  * 60 + dropoff_time * 60
        
        # Update the driver's information
        driver['cluster'] = passenger_destination
        driver['Hour'] = passenger['Hour']
        driver['DayType'] = passenger['DayType']
        driver['Date/Time'] = new_driver_time
        driver['number_of_trips'] += 1
        
        # Add this trip to the matches list
        matches.append({
            'driver_location': driver_location,
            'passenger_pickup': passenger_pickup,
            'passenger_destination': passenger_destination,
            'pickup_wait_time': travel_to_pickup_time,
            'dropoff_time': dropoff_time,
            'wait_from_passenger_request': wait_from_passenger_request,
            'total_wait': travel_to_pickup_time + dropoff_time + wait_from_passenger_request,
        })
        
        # simulate drivers stopping to drive
        if driver['number_of_trips'] > 10 and len(driver_queue) > 20:
            random_number = random.randint(1, 10)

            # 1/10 chance of driver stopping after 10 trips
            if random_number == 1:
                heapq.heappush(driver_queue, (new_driver_time, id, driver))
            else:
                exited_drivers.append(driver)
        else:
            # Re-insert the driver into the priority queue with the new available time
            heapq.heappush(driver_queue, (new_driver_time, id, driver))
        
        total_time_drivers_travel_to_passengers += travel_to_pickup_time
        total_in_car_time += dropoff_time
        

        if len(passenger_queue) % 100 == 0:
            print(len(passenger_queue), 'passengers in queue')
            print(len(driver_queue), 'drivers in queue')



    trips_per_driver = []
    all_drivers = [driver for _, _, driver in driver_queue] + exited_drivers
    for driver in all_drivers:
        trips_per_driver.append(driver['number_of_trips'])
        
    return matches


# T5, locally optimal choice for driver with parallelization
def simulate_t5(graph, passenger_queue, driver_queue):
    print('Running T5 simulation...')
    matches = []  # Track every trip
    total_time_drivers_travel_to_passengers = 0
    total_in_car_time = 0
    exited_drivers = []
    num_processes = multiprocessing.cpu_count()

    # passenger_queue = passenger_queue[-200:]

    
    while passenger_queue:  # Continue until one of the queues is empty
        # Passenger and driver details

        driver_available_time, driver_id, driver = heapq.heappop(driver_queue)


        # simulate higher paying customer who does not want to wait
        priority_customer = random.randint(1, 25) == 1
  
        if not priority_customer:
        # passenger_request_time, _, passenger = passenger_queue.pop()  
            availible_passengers = []

            while passenger_queue and passenger_queue[-1][0] >= driver_available_time and len(availible_passengers) < 6:
                availible_passengers.append(passenger_queue.pop())

            if len(availible_passengers) == 0:
                availible_passengers.append(passenger_queue.pop())

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
                # Parallelize the Dijkstra calculations for available drivers
                results = list(executor.map(lambda p: calculate_trip_ratio(p[2], driver['node'], driver, graph), availible_passengers))

            
            _, travel_to_pickup_time, dropoff_time, optimal_passenger = optimal_with_density_penalty(results, driver_available_time)

            availible_passengers.sort(key=lambda x: x[0], reverse=True)

            for passenger_request_time, _, passenger in availible_passengers:
                if passenger != optimal_passenger:
                    passenger_queue.append((passenger_request_time, _, passenger))

        else:
            passenger_request_time, _, optimal_passenger = passenger_queue.pop()
            travel_to_pickup_time = dijkstra(graph, driver['node'], optimal_passenger['node'], driver['Hour'], driver['DayType'])
            dropoff_time = dijkstra(graph, optimal_passenger['node'], optimal_passenger['destination_node'], driver['Hour'], driver['DayType'])


        wait_from_passenger_request = 0
        if passenger_request_time < driver_available_time:
            wait_from_passenger_request = (driver_available_time - passenger_request_time) / 60
                
        # Calculate the driver's new available time
        new_driver_time = driver_available_time + travel_to_pickup_time  * 60 + dropoff_time * 60
        
        driver_location = driver['node']
        # Update the driver's information
        driver['node'] = optimal_passenger['destination_node']
        driver['Hour'] = optimal_passenger['Hour']
        driver['DayType'] = optimal_passenger['DayType']
        driver['Date/Time'] = new_driver_time
        driver['number_of_trips'] += 1
        
        # Add this trip to the matches list
        matches.append({
            'driver_location': driver_location,
            'passenger_pickup': optimal_passenger['node'],
            'passenger_destination': optimal_passenger['destination_node'],
            'pickup_wait_time': travel_to_pickup_time,
            'dropoff_time': dropoff_time,
            'wait_from_passenger_request': wait_from_passenger_request,
            'total_wait': travel_to_pickup_time + dropoff_time + wait_from_passenger_request,
            'priority_customer': priority_customer
        })
        
        # simulate drivers stopping to drive
        if driver['number_of_trips'] > 10 and len(driver_queue) > 20:
            random_number = random.randint(1, 10)

            # 1/10 chance of driver stopping after 10 trips
            if random_number == 1:
                heapq.heappush(driver_queue, (new_driver_time, id, driver))
            else:
                exited_drivers.append(driver)
        else:
            # Re-insert the driver into the priority queue with the new available time
            heapq.heappush(driver_queue, (new_driver_time, id, driver))
        
        total_time_drivers_travel_to_passengers += travel_to_pickup_time
        total_in_car_time += dropoff_time
        

        if len(passenger_queue) % 100 == 0:
            print(len(passenger_queue), 'passengers in queue')


    trips_per_driver = []
    all_drivers = [driver for _, _, driver in driver_queue] + exited_drivers
    for driver in all_drivers:
        trips_per_driver.append(driver['number_of_trips'])
        

    return matches


# T5 with multiprocesssors for parallelization
def simulate_t5_multiprocesssors(graph, passenger_queue, driver_queue):
    print('Running T5 simulation in prll...')
    matches = []  # Track every trip
    total_time_drivers_travel_to_passengers = 0
    total_in_car_time = 0
    failute_count = 0
    exited_drivers = []

    passenger_queue = passenger_queue[-200:]
    num_processes = multiprocessing.cpu_count()
    
    while passenger_queue:  # Continue until one of the queues is empty
        # Passenger and driver details
        passenger_request_time, _, passenger = passenger_queue.pop()  


        available_drivers = []

        while driver_queue and driver_queue[0][0] > passenger_request_time and len(available_drivers) < num_processes :
            available_drivers.append(heapq.heappop(driver_queue))

        # Pop all available drivers whose availability time is less than or equal to the passenger request time

        driver = None
        given_travel_to_pickup_time = None

        if available_drivers:

            pool = multiprocessing.Pool(processes=num_processes//3)

            # Parallelize the Dijkstra calculations for available drivers
            driver_distances = pool.starmap(calculate_driver_distances, [(driver, passenger['node'], graph) for driver in available_drivers])
            
            # Close the pool of processes
            pool.close()
            pool.join()

            driver_distances.sort()
            # Select the driver with the shortest time to get there
            shortest_travel_to_pickup_time, driver_time, id, driver = driver_distances[0]
            given_travel_to_pickup_time = shortest_travel_to_pickup_time
            
            # Re-insert the other drivers into the priority queue
            for d_time, id, d in available_drivers:
                if d != driver:
                    heapq.heappush(driver_queue, (d_time, id, d))

        else:
            # If no drivers are available yet, take the earliest available driver
            driver_time, id, driver = heapq.heappop(driver_queue)

        ###
        # Get the driver's current location and passenger's pickup location
        driver_location = driver['node']
        passenger_pickup = passenger['node']
        
        # Calculate time from passenger making request to driver becoming available
        wait_from_passenger_request = 0
        if passenger_request_time < driver_time:
            wait_from_passenger_request = (driver_time - passenger_request_time) / 60



        # Calculate time for driver to reach passenger, unless we have already computed it
        if given_travel_to_pickup_time is None:
            travel_to_pickup_time = dijkstra(graph, driver_location, passenger_pickup, driver['Hour'], driver['DayType'])
        else:
            travel_to_pickup_time = given_travel_to_pickup_time
        
        if travel_to_pickup_time == float('inf'):
            print('No path to passenger', passenger, driver)
            failute_count += 1
            continue

        
        # Calculate time for driver to drop passenger at the destination
        passenger_destination = passenger['destination_node']
        dropoff_time = dijkstra(graph, passenger_pickup, passenger_destination, passenger['Hour'], passenger['DayType'])

        if dropoff_time == float('inf'):
            print('No path to destination', passenger, driver)
            failute_count += 1
            continue
        
        # Calculate the driver's new available time
        new_driver_time = driver_time + travel_to_pickup_time  * 60 + dropoff_time * 60
        
        # Update the driver's information
        driver['node'] = passenger_destination
        driver['Hour'] = passenger['Hour']
        driver['DayType'] = passenger['DayType']
        driver['Date/Time'] = new_driver_time
        driver['number_of_trips'] += 1
        
        # Add this trip to the matches list
        matches.append({
            'driver_location': driver_location,
            'passenger_pickup': passenger_pickup,
            'passenger_destination': passenger_destination,
            'pickup_wait_time': travel_to_pickup_time,
            'dropoff_time': dropoff_time,
            'wait_from_passenger_request': wait_from_passenger_request,
            'total_wait': travel_to_pickup_time + dropoff_time + wait_from_passenger_request,
        })
        
        # simulate drivers stopping to drive
        if driver['number_of_trips'] > 10 and len(driver_queue) > 20:
            random_number = random.randint(1, 10)

            # 1/10 chance of driver stopping after 10 trips
            if random_number == 1:
                heapq.heappush(driver_queue, (new_driver_time, id, driver))
            else:
                exited_drivers.append(driver)
        else:
            # Re-insert the driver into the priority queue with the new available time
            heapq.heappush(driver_queue, (new_driver_time, id, driver))
        
        total_time_drivers_travel_to_passengers += travel_to_pickup_time
        total_in_car_time += dropoff_time
        

        if len(passenger_queue) % 100 == 0:
            print(len(passenger_queue), 'passengers in queue')
            print(len(driver_queue), 'drivers in queue')


    trips_per_driver = []
    all_drivers = [driver for _, _, driver in driver_queue] + exited_drivers
    for driver in all_drivers:
        trips_per_driver.append(driver['number_of_trips'])
        

    return matches


# Compute dependencies and run simulation
def wrapper(given_simulation = 'T1', reprocess_data=False, rebuild_graph=False):
    start_time = time.time()

    # Build graph
    if rebuild_graph:
       build_graph()
    
    # Load graph
    graph = load_graph()


    if reprocess_data:
        # Read in un-proccessed data
        drivers_data, passengers_data = read_base_data()

        # Pre-process data
        if given_simulation == 'T4_A':
            kd_tree_pre_processing(graph, drivers_data, passengers_data)
            # print time to process with kd tree
            print('Time to process with kd tree:', time.time() - start_time)
        elif given_simulation == 'T4_B':
            tiny_graph_pre_processing(graph, drivers_data, passengers_data)
            # print time to process with tiny graph
            print('Time to process with tiny graph:', time.time() - start_time)
        else:
            simple_pre_processing(graph, drivers_data, passengers_data)
            # print time to process with simple
            print('Time to process with simple:', time.time() - start_time)

    # Load pre-processed data
    if given_simulation == 'T4_B':
          drivers_data, passengers_data = load_tiny_graph_updated_data()
    elif given_simulation == 'T4_A':
        drivers_data, passengers_data = load_kd_updated_data()
    else:
        drivers_data, passengers_data = load_updated_data()

    # Construct queues
    passenger_queue, driver_queue = construct_queues(drivers_data, passengers_data)

    # Run simulation
    if given_simulation == 'T1':
        matches = simulate_t1(graph, passenger_queue, driver_queue)
    elif given_simulation == 'T2':
        matches = simulate_t2(graph, passenger_queue, driver_queue)
    elif given_simulation == 'T3':
        matches = simulate_t3(graph, passenger_queue, driver_queue)
    elif given_simulation == 'T4':
        matches= simulate_t4(passenger_queue, driver_queue)
    elif given_simulation == 'T5':
        matches = simulate_t5(graph, passenger_queue, driver_queue)
    else:
        print('Invalid simulation')
        return
    
    # print time to run simulation
    print(f'Time to run simulation {given_simulation}:', time.time() - start_time)

    # Write results to file
    if given_simulation == 'T1':
        results_file = 'T1_extra.json'
    elif given_simulation == 'T2':
        results_file = 'T2_extra.json'
    elif given_simulation == 'T3':
        results_file = 'T3_extra.json'
    elif given_simulation == 'T4':
        results_file = 'T4_extra.json'
    elif given_simulation == 'T5':
        results_file = 'T5_extra.json'

    with open(results_file, 'w') as f:
        json.dump(matches, f, indent=4)



if __name__ == "__main__":
    wrapper('T5', False, False)