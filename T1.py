import csv
from datetime import datetime
from collections import deque
import json
from math import radians, cos, sin, asin, sqrt


# READ IN PASSENGER/DRIVER DATA
def read_csv_file(filepath):
    data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

drivers_filepath = 'drivers.csv'
passengers_filepath = 'passengers.csv'

drivers_data = read_csv_file(drivers_filepath)
passengers_data = read_csv_file(passengers_filepath)

print(drivers_data[100], passengers_data[0])

# READ IN NODE DATA
# Function to read a JSON file
def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

# Reading the node data and connection data from JSON files
node_coordinates_filepath = 'node_data.json'
node_connections_filepath = 'adjacency.json'

node_coordinates = read_json_file(node_coordinates_filepath)
node_connections = read_json_file(node_connections_filepath)


# Building the graph
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
        # Ensure both nodes exist in the graph before adding the connection
        if start_node_id in graph and end_node_id in node_coordinates:
            graph[start_node_id]['connections'][end_node_id] = attributes

# At this point, `graph` is a dictionary representing the graph structure,
# where each node has 'coordinates' and 'connections' to other nodes with specific attributes.

print(graph)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points
    on the Earth surface given their longitude and latitude in degrees.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of Earth in kilometers. Use 3956 for miles.
    return c * r


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



# for each node in drivers_data, find the nearest node in graph and add it to the dictionary
for driver in drivers_data:
    driver['node'] = find_nearest_node(graph, float(driver['Source Lat']), float(driver['Source Lon']))

# for each node in passengers_data, find the nearest node in graph and add it to the dictionary
for passenger in passengers_data:
    passenger['node'] = find_nearest_node(graph, float(passenger['Source Lat']), float(passenger['Source Lon']))




# Function to parse the datetime string
def parse_datetime(datetime_str):
    return datetime.datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')

import heapq
passenger_queue = [(p['Date/Time'], p) for p in passengers]
heapq.heapify(passenger_queue)

driver_queue = [(d['available_since'], d) for d in drivers]
heapq.heapify(driver_queue)

# Function to find a driver for the longest waiting passenger
def assign_driver(passengers, drivers):
    if not passengers or not drivers:
        return None, None  # No assignment possible

    # Find the longest waiting passenger
    longest_waiting_passenger = passengers.popleft()
    
    # Assign the first available driver
    assigned_driver = drivers.popleft()
    
    return longest_waiting_passenger, assigned_driver


# def calculate_driving_time(graph, from_node, to_node, current_time):
#     # Lookup the edge in the graph
#     # Here you would consider 'current_time' to check 'day_type' and 'hour'
#     # and then calculate the driving time based on 'length' and 'max_speed'
#     # This is a placeholder function, you'll need to implement the actual logic
#     return graph[from_node]['connections'][to_node]['time']


# # Function to assign drivers to passengers
# def assign_drivers(graph, passenger_queue, driver_queue):
#     matches = []

#     while passenger_queue and driver_queue:
#         # Get the passenger who has been waiting the longest
#         passenger_wait_time, passenger = heapq.heappop(passenger_queue)

#         # Find the closest available driver
#         closest_driver = None
#         closest_time = float('inf')
#         for driver_wait_time, driver in driver_queue:
#             driving_time = calculate_driving_time(graph, driver['node'], passenger['node'], passenger['Date/Time'])
#             if driving_time < closest_time:
#                 closest_driver = driver
#                 closest_time = driving_time
        
#         if closest_driver:
#             # Assign the driver to the passenger
#             matches.append((passenger, closest_driver))
            
#             # Simulate the driver's trip and update the driver's availability time
#             # For now, we'll just add a fixed amount of time for simplicity
#             closest_driver['available_since'] = passenger['Date/Time'] + datetime.timedelta(minutes=closest_time + 30)
            
#             # Push the driver back into the driver queue
#             heapq.heappush(driver_queue, (closest_driver['available_since'], closest_driver))

#     return matches

# # Run the assignment algorithm
# matches = assign_drivers(graph, passenger_queue, driver_queue)

# # Output the results
# for passenger, driver in matches:
#     print(f"Passenger at node {passenger['node']} matched with driver at node {driver['node']}")