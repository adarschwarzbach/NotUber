import csv
from datetime import datetime
from collections import deque
import json
from math import radians, cos, sin, asin, sqrt
import heapq
from datetime import datetime, timedelta




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

# print(graph)

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


# print('Finding nearest node for each driver...')
# # for each node in drivers_data, find the nearest node in graph and add it to the dictionary
# for driver in drivers_data:
#     driver['node'] = find_nearest_node(graph, float(driver['Source Lat']), float(driver['Source Lon']))


# print('Finding nearest node for each passenger...')
# # for each node in passengers_data, find the nearest node in graph and add it to the dictionary
# for passenger in passengers_data:
#     passenger['node'] = find_nearest_node(graph, float(passenger['Source Lat']), float(passenger['Source Lon']))
#     passenger['destination_node'] = find_nearest_node(graph, float(passenger['Dest Lat']), float(passenger['Dest Lon']))


drivers_file = 'updated_drivers.json'
passengers_file = 'updated_passengers.json'


# print('Writing updated driver data to file...')
# with open(drivers_file, 'w') as f:
#     json.dump(drivers_data, f, indent=4)  # Use indent=4 for pretty-printing

# print('Writing updated passenger data to file...')
# with open(passengers_file, 'w') as f:
#     json.dump(passengers_data, f, indent=4)



with open(drivers_file, 'r') as f:
    drivers_data = json.load(f)

with open(passengers_file, 'r') as f:
    passengers_data = json.load(f)


# Function to parse the datetime string
def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')


# Create a priority queue for passengers with a secondary sort key
passenger_queue = [(parse_datetime(p['Date/Time']), id(p), p) for p in passengers_data]
heapq.heapify(passenger_queue)

# Create a priority queue for drivers with a secondary sort key
driver_queue = [(parse_datetime(d["Date/Time"]), id(d), d) for d in drivers_data]
heapq.heapify(driver_queue)


print(' \n Starting to match drivers and passengers... \n')

def dijkstra(graph, start, end):
    # Priority queue to hold the nodes to visit next, initialized with the start node
    queue = [(0, start)]  # (cumulative_time, node)
    visited = set()
    while queue:
        # Get the node with the smallest cumulative time
        cumulative_time, node = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)
            if node == end:
                # We've reached the destination
                return cumulative_time
            # Visit all neighbors of the current node
            for neighbor, edge in graph[node]['connections'].items():
                if neighbor not in visited:
                    # Add the travel time for this edge to the cumulative time
                    travel_time = edge['time'] * 60  # Convert hours to minutes
                    heapq.heappush(queue, (cumulative_time + travel_time, neighbor))
    # If the destination is not reachable, return infinity
    return float('inf')




def calculate_driving_time(graph, from_node, to_node, current_time=0):
    """
    Calculates the driving time between two nodes using Dijkstra's algorithm.
    
    :param graph: The graph data containing nodes and connections.
    :param from_node: The starting node ID as a string.
    :param to_node: The destination node ID as a string.
    :param current_time: The current time as a datetime object.
    :return: Driving time in minutes as a float.
    """
    # Run Dijkstra's algorithm to find the shortest path
    return dijkstra(graph, from_node, to_node)



def assign_drivers(graph, passenger_queue, driver_queue):
    matches = []

    # We will track the time spent driving to pick up passengers and the time driving passengers to drop-off
    total_pickup_time = 0
    total_ride_time = 0

    while passenger_queue and driver_queue:
        # Get the passenger who has been waiting the longest
        passenger_wait_time, _, passenger = heapq.heappop(passenger_queue)  # Adjusted here

        # Find the closest available driver using the graph data
        closest_driver = None
        closest_driver_time = None
        shortest_pickup_time = float('inf')
        closest_driver_index = -1
        for i, (driver_available_since, _, driver) in enumerate(driver_queue):  # Adjusted here
            # Calculate driving time to the passenger's pickup location
            pickup_time = calculate_driving_time(graph, driver['node'], passenger['node'])
            
            if pickup_time < shortest_pickup_time:
                closest_driver = driver
                closest_driver_time = driver_available_since  # Already a datetime object
                shortest_pickup_time = pickup_time
                closest_driver_index = i

        if closest_driver_index >= 0:  # Check if we found a driver
            # Assign the driver to the passenger
            matches.append((passenger, closest_driver))
            total_pickup_time += shortest_pickup_time
            
            # Calculate the drop-off time using the graph data
            ride_time = calculate_driving_time(graph, passenger['node'], passenger['destination_node'])
            total_ride_time += ride_time

            # Update the driver's next available time based on the ride time
            driver_dropoff_time = closest_driver_time + timedelta(minutes=(shortest_pickup_time + ride_time))
            driver_queue[closest_driver_index] = (driver_dropoff_time, id(closest_driver), closest_driver)
            
            # Since we've modified a queue element, we need to re-heapify
            heapq.heapify(driver_queue)

    # Calculate profits (D2) as total ride time minus time spent for pickups
    total_profit = total_ride_time - total_pickup_time

    return matches, total_profit, total_pickup_time, total_ride_time

# Run the assignment algorithm
matches, total_profit, total_pickup_time, total_ride_time = assign_drivers(graph, passenger_queue, driver_queue)

# Output the results
for match in matches:
    print(f"Passenger {match[0]['node']} picked up by Driver {match[1]['node']}")
print(f"Total Profit: {total_profit} minutes")
print(f"Total Pickup Time: {total_pickup_time} minutes")
print(f"Total Ride Time: {total_ride_time} minutes")