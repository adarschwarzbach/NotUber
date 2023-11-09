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
        if start_node_id in graph and end_node_id in node_coordinates:
            graph[start_node_id]['connections'][end_node_id] = attributes



json_string = json.dumps(graph, indent=4)
print(json_string)


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


def parse_datetime_to_unix(datetime_str):
    try:
        # Convert the datetime string to a datetime object
        dt = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
        # Convert the datetime object to a Unix timestamp
        return int(dt.timestamp())
    except ValueError as e:
        # Handle the error: log it, return None, or use a default value
        print(f"Error parsing datetime: {e}")
        return None

# Convert Date/Time for all passengers and drivers to Unix timestamps within the data
for p in passengers_data:
    unix_time = parse_datetime_to_unix(p['Date/Time'])
    if unix_time is not None:
        p['Date/Time'] = unix_time

for d in drivers_data:
    unix_time = parse_datetime_to_unix(d['Date/Time'])
    if unix_time is not None:
        d['Date/Time'] = unix_time

# Construct queues
passenger_queue = [(p['Date/Time'], p) for p in passengers_data if 'Date/Time' in p]

driver_queue = [(d['Date/Time'], d) for d in drivers_data if 'Date/Time' in d]
heapq.heapify(driver_queue)


print(' \n Starting to match drivers and passengers... \n')


def dijkstra(graph, start, end):
    queue = [(0, start)]  # (cumulative_time, node)
    visited = set()
    # Map to store shortest distance to a node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while queue:
        # Get the node with the smallest cumulative time
        cumulative_time, node = heapq.heappop(queue)
        if node not in visited:
            visited.add(node)

            # Return if we have reached the destination
            if node == end:
                return cumulative_time

            for neighbor, edge in graph[node]['connections'].items():
                travel_time = edge['time'] * 60  # Convert hours to minutes
                new_time = cumulative_time + travel_time


                if new_time < distances[neighbor]:
                    distances[neighbor] = new_time
                    heapq.heappush(queue, (new_time, neighbor))

    # If the destination is not reachable, return infinity
    return float('inf')


print(passenger_queue[0])

def simulate(graph, passenger_queue, driver_queue):
    matches = []
    total_pickup_time = 0
    total_wait_time = 0  # D1: Total wait time for all passengers


    pass


# def assign_drivers(graph, passenger_queue, driver_queue):
#     matches = []

#     # We will track the time spent driving to pick up passengers
#     total_pickup_time = 0
#     total_wait_time = 0  # D1: Total wait time for all passengers

#     while passenger_queue:
#         # Get the passenger who has been waiting the longest
#         passenger_wait_time, _, passenger = heapq.heappop(passenger_queue)

#         if not driver_queue:
#             # If there are no drivers available, we break out of the loop
#             break
        
#         # Get the first available driver
#         driver_available_since, _, driver = heapq.heappop(driver_queue)

#         # Calculate driving time to the passenger's pickup location using Dijkstra's algorithm
#         pickup_time = calculate_driving_time(graph, driver['node'], passenger['node'])

#         if pickup_time == float('inf'):
#             # If there is no path to the passenger, we cannot service this passenger.
#             continue  # Skip to the next passenger

#         # Calculate wait time for the passenger
#         # This is the current time plus the time it takes for the driver to reach them minus the time they started waiting
#         now = datetime.now()
#         wait_time_minutes = (now + timedelta(minutes=pickup_time) - passenger_wait_time).total_seconds() / 60
#         total_wait_time += wait_time_minutes
#         total_pickup_time += pickup_time

#         # Store the passenger-driver match with the wait time
#         matches.append((passenger, driver, wait_time_minutes))

#         ride_time = calculate_driving_time(graph, passenger['node'], passenger['destination_node'])

#         # Update the driver's next available time based on the ride time
#         driver['available_since'] = now + timedelta(minutes=(pickup_time + ride_time))

#         # Re-add the driver to the driver queue with the updated available time
#         heapq.heappush(driver_queue, (driver['available_since'], id(driver), driver))

#     # D1 is the average wait time, which we can calculate by dividing the total wait time by the number of matches
#     average_wait_time = total_wait_time / len(matches) if matches else 0
#     # D2 and D3 are not calculated here as they require more information

#     return matches, total_pickup_time, average_wait_time


# if __name__ == "__main__":
#     pass