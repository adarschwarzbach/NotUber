import csv
from datetime import datetime
from collections import deque
import json


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

pretty_json = json.dumps(graph, indent=4)
print(graph['22955061'])