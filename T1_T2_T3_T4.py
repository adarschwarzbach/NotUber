import csv
from datetime import datetime
from collections import deque
import json
from math import radians, cos, sin, asin, sqrt
import heapq
from datetime import datetime, timedelta
import random




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


# Load pre-processed data
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
                travel_time = edge * 60 * 7.42   # Convert hours to minutes and multiply by 7.42 as there are on average 7.42 fewer edges in a given trip
                new_time = cumulative_time + travel_time
                new_stops = num_stops + 1

                if new_time < distances[neighbor][0]:
                    distances[neighbor] = (new_time, new_stops)
                    heapq.heappush(queue, (new_time, neighbor, new_stops))

    # If the destination is not reachable, return infinity for both time and stops
    return float('inf'), float('inf')





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
    passenger_queue = passenger_queue[1000:1200]
    
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
    return matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver, 


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
        

    return matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver, 


# When there is a choice, passenger will be assigned to the driver with the shortest time to drive to them
def simulate_t3(graph, passenger_queue, driver_queue):
    print('Running T3 simulation...')
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
        

    return matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver, 





# Optimize T3: use approximate times with the tiny graph of clusters
def simulate_t4_b(passenger_queue, driver_queue):
    print('Running T4 B simulation...')

    with open('graph.json', 'r') as f:
        g = json.load(f)
        print('g', len(g))
    # load tiny_graph.json
    with open('tiny_graph.json', 'r') as f:
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
        
    return matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver, 







# Compute dependencies and run simulation
def wrapper(given_simulation = 'T1', reprocess_data=False, rebuild_graph=False):

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
    if given_simulation == 'T4_B':
          drivers_data, passengers_data = load_tiny_graph_updated_data()
    else:
        drivers_data, passengers_data = load_updated_data()

    # Construct queues
    passenger_queue, driver_queue = construct_queues(drivers_data, passengers_data)

    # Run simulation
    if given_simulation == 'T1':
        matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver = simulate_t1(graph, passenger_queue, driver_queue)
    elif given_simulation == 'T2':
        matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver = simulate_t2(graph, passenger_queue, driver_queue)
    elif given_simulation == 'T3':
        matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver = simulate_t3(graph, passenger_queue, driver_queue)
    elif given_simulation == 'T4_B':
        matches, total_time_drivers_travel_to_passengers, total_in_car_time, wait_from_passenger_request, failute_count, trips_per_driver = simulate_t4_b(passenger_queue, driver_queue)

    # Write results to file
    if given_simulation == 'T1':
        results_file = 'T1_extra.json'
    elif given_simulation == 'T2':
        results_file = 'T2_extra.json'
    elif given_simulation == 'T3':
        results_file = 'T3_extra.json'
    elif given_simulation == 'T4_B':
        results_file = 'T4_B_extra.json'

    with open(results_file, 'w') as f:
        json.dump(matches, f, indent=4)



if __name__ == "__main__":
    wrapper('T4_B', False, False)