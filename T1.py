import csv
from datetime import datetime
from collections import deque
import json
from math import radians, cos, sin, asin, sqrt
import heapq
from datetime import datetime, timedelta
from collections import defaultdict




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
             graph[end_node_id]['connections'][start_node_id] = attributes

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


# Load pre-processed data
def load_updated_data():
    drivers_file = 'updated_drivers.json'
    passengers_file = 'updated_passengers.json'

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
        # Convert the datetime string to a datetime object
        dt = datetime.strptime(datetime_str, '%m/%d/%Y %H:%M:%S')
        # Determine if the day is a weekday or weekend
        day_type = 'weekday' if dt.weekday() < 5 else 'weekend'
        # Extract the hour of the day
        hour = dt.hour
        # Convert the datetime object to a Unix timestamp
        unix_timestamp = int(dt.timestamp())
        return unix_timestamp, hour, day_type
    except ValueError as e:
        # Handle the error: log it, return None, or use a default value
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
    heapq.heapify(driver_queue)

    print(passenger_queue[0], driver_queue[0])
    return passenger_queue, driver_queue



# Djikstra's implementation to determine time for a given trip
# ToDo: Update hour based on traversal time
def dijkstra(graph, start, end, hour, day_type):
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

                if new_time < distances[neighbor]:
                    distances[neighbor] = new_time
                    heapq.heappush(queue, (new_time, neighbor))

    # If the destination is not reachable, return infinity
    return float('inf')


#Update to take into account idle time
# Main function to run simulation
def simulate(graph, passenger_queue, driver_queue):
    matches = []  # Track every trip
    total_time_drivers_travel_to_passengers = 0
    total_in_car_time = 0
    failute_count = 0

    #test on smaller queue
    # passenger_queue = passenger_queue[1000:1200]
    
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
        
        # Update the driver's location to the passenger's destination
        driver['node'] = passenger_destination

        # update the driver hour and day type
        driver['Hour'] = passenger['Hour']
        driver['DayType'] = passenger['DayType']

        # update driver time 
        driver['Date/Time'] = new_driver_time

        # update number of trips
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
        
        # Re-insert the driver into the priority queue with the new available time
        heapq.heappush(driver_queue, (new_driver_time, id(driver), driver))
        
        total_time_drivers_travel_to_passengers += travel_to_pickup_time
        total_in_car_time += dropoff_time
        

        if len(passenger_queue) % 100 == 0:
            print(len(passenger_queue))


    trips_per_driver = []
    for driver in driver_queue:
        trips_per_driver.append(driver[2]['number_of_trips'])


    return matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver


# Compute dependencies and run simulation
def wrapper(reprocess_data=False, rebuild_graph=False):

    # Build graph
    if rebuild_graph:
       build_graph()
    
    # Load graph
    graph = load_graph()
    
    # pretty_graph = json.dumps(graph, indent=4)
    # print(pretty_graph)

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
    matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver = simulate(graph, passenger_queue, driver_queue)

    # Print results
    print(f"Total failures: {failute_count}")
    print(f"Total pickup time: {total_time_drivers_travel_to_passengers}")
    print(f"Total in car time: {total_in_car_time}")
    print(f"Average total trip time: {(total_time_drivers_travel_to_passengers +  total_in_car_time + wait_from_passenger_request)/ len(matches)}")
    print(f"Average number of trips per driver: {sum(trips_per_driver)/len(trips_per_driver)}")
    print(f"Number of drivers with zero trips: {len([driver for driver in trips_per_driver if driver == 0])}")

    # Write results to file
    results_file = 'T1_results.json'
    with open(results_file, 'w') as f:
        json.dump(matches, f, indent=4)




if __name__ == "__main__":
    wrapper(False, False)