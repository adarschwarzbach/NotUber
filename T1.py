import csv
from datetime import datetime
from collections import deque
import json
from math import radians, cos, sin, asin, sqrt
import heapq
from datetime import datetime, timedelta




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
            if start_node_id in graph and end_node_id in node_coordinates:
                graph[start_node_id]['connections'][end_node_id] = attributes

    return graph



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


# Load pre-processed data
def load_updated_data():
    drivers_file = 'updated_drivers.json'
    passengers_file = 'updated_passengers.json'

    with open(drivers_file, 'r') as f:
        drivers_data = json.load(f)

    with open(passengers_file, 'r') as f:
        passengers_data = json.load(f)

    return drivers_data, passengers_data


# Helper to convert a datetime string to a Unix timestamp
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


# Build passenger queue and driver priority queue
def construct_queues(drivers_data, passengers_data):
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
    # Use index as a secondary sort key to ensure dictionaries are not compared
    passenger_queue = [(p['Date/Time'], i, p) for i, p in enumerate(passengers_data) if 'Date/Time' in p]
    passenger_queue = passenger_queue[::-1]  # Reverse the list so that the earliest passengers are at the front

    driver_queue = [(d['Date/Time'], i, d) for i, d in enumerate(drivers_data) if 'Date/Time' in d]
    heapq.heapify(driver_queue)

    return passenger_queue, driver_queue



# Djikstra's implementation to determine time for a given trip
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



# Main function to run simulation
def simulate(graph, passenger_queue, driver_queue):
    matches = []  # Track every trip
    total_time_drivers_travel_to_passengers = 0
    total_in_car_time = 0
    failute_count = 0
    

    while passenger_queue:  # Continue until one of the queues is empty
        # Passenger and driver details
        _, _, passenger = passenger_queue.pop(0)  # Pop from the front of the list (FIFO)
        driver_time, _, driver = heapq.heappop(driver_queue)  # Pop the first available driver
        
        # Get the driver's current location and passenger's pickup location
        driver_location = driver['node']
        passenger_pickup = passenger['node']
        
        # Calculate time for driver to reach passenger
        travel_to_pickup_time = dijkstra(graph, driver_location, passenger_pickup)
        
        if travel_to_pickup_time == float('inf'):
            print('No path to passenger', passenger, driver)
            failute_count += 1
            continue

        
        # Calculate time for driver to drop passenger at the destination
        passenger_destination = passenger['destination_node']
        dropoff_time = dijkstra(graph, passenger_pickup, passenger_destination)

        if dropoff_time == float('inf'):
            print('No path to destination', passenger, driver)
            failute_count += 1
            continue
        
        # Calculate the driver's new available time
        new_driver_time = driver_time + travel_to_pickup_time + dropoff_time
        
        # Update the driver's location to the passenger's destination
        driver['node'] = passenger_destination
        
        # Add this trip to the matches list
        matches.append({
            'driver_location': driver_location,
            'passenger_pickup': passenger_pickup,
            'passenger_destination': passenger_destination,
            'pickup_wait_time': travel_to_pickup_time,
            'dropoff_time': dropoff_time,
            'total_wait': travel_to_pickup_time + dropoff_time,
        })
        
        # Re-insert the driver into the priority queue with the new available time
        heapq.heappush(driver_queue, (new_driver_time, id(driver), driver))
        
        total_time_drivers_travel_to_passengers += travel_to_pickup_time
        total_in_car_time += dropoff_time
        
        print(len(passenger_queue))


    return matches, total_time_drivers_travel_to_passengers, total_in_car_time, failute_count

# Compute dependencies and run simulation
def wrapper(reprocess_data=False):

    # Build graph
    graph = build_graph()

    if reprocess_data:
        # Read in un-proccessed data
        drivers_data, passengers_data = read_base_data()

        # Pre-process data
        simple_pre_processing(graph, drivers_data, passengers_data)

    # Load pre-processed data
    drivers_data, passengers_data = load_updated_data()

    # Construct queues
    passenger_queue, driver_queue = construct_queues(drivers_data, passengers_data)

    # Run simulation
    matches, total_time_drivers_travel_to_passengers, total_in_car_time, failute_count = simulate(graph, passenger_queue, driver_queue)

    # Print results
    print(f"Total failures: {failute_count}")
    print(f"Total pickup time: {total_time_drivers_travel_to_passengers}")
    print(f"Total in car time: {total_in_car_time}")
    print(f"Average total trip time: {(total_time_drivers_travel_to_passengers +  total_in_car_time)/ len(matches)}")




if __name__ == "__main__":
    wrapper()