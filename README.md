# NotUber - Case Study

## Introduction
Welcome to the Preface Case Study! As newly graduated computer scientists from Duke University, we are the "Data-Driven Algorithms Design Group" at NotUber (NU). Our mission is to design an algorithmic system to automate real-time driver-passenger assignments in New York City.

## Problem Formulation
We focus on an undirected graph representing NYC's road network, where passengers and drivers interact through our mobile application. The challenge is to develop a system that effectively matches drivers to passengers while optimizing time efficiency, ride profit, and scalability.

## Getting Started
Before diving into the algorithm design, ensure you have Python 3 installed as it is our primary language for this project.

### Prerequisites
- Python 3.x
- Standard library modules only, no external packages required.

## Event-Driven Decisions
Our algorithm makes decisions based on events such as new passenger requests or drivers becoming available. The goal is to respond promptly and efficiently to these events, balancing our desiderata:

- **D1**: Minimize wait time for passengers.
- **D2**: Maximize ride profit for drivers.
- **D3**: Ensure the empirical efficiency and scalability of our algorithms.

## Constraints
We operate under the following constraints:

- **C1**: One passenger per driver.
- **C2**: Strict pickup and drop-off locations.
- **C3**: Irrevocable matches.

## Algorithm Design and Implementation
The repository includes several baseline algorithms and improvements:

- **T1**: Simple queue-based matching system.
- **T2**: Proximity-based driver assignment.
- **T3**: Network traversal time optimization.
- **T4**: Runtime efficiency improvements.
- **T5**: Proprietary algorithm for enhanced scheduling.