# NotUber

Simulation of a ride-hailing system :)

## Features

### Simple Queue-Based Matching
- Matches riders and drivers using a basic FIFO queue system.

### Proximity-Based Driver Assignment
- Matches riders with the nearest available driver using shortest path algorithms.

### Network Traversal Time Optimization
- Optimizes the travel time between riders and drivers

### Efficiency Improvements
- Utilizes KD-Tree and hypergraph to decrease empirical runtime 100x

### Proprietary Enhanced Scheduling
- T4 + additional heurestics for drivers to select rides while considering future implications of drop-off location
