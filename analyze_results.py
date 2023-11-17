import json

#  T1_results.json | T1_extra.txt | T2_results.json | T2_extra.txt | T3_results.json | T3_extra.json | T5_results.json | T5_extra.json
file = 'T5_extra.json'
with open(file, 'r') as f:
    T1_results = json.load(f)

print(f"-- {file[:-5]} Results--")
average_waiting_time = 0
for result in T1_results:
    average_waiting_time += result['total_wait']

average_waiting_time /= len(T1_results)

print('Average trip time: ', average_waiting_time)


number_of_trips_over_100 = 0
for result in T1_results:
    if result['total_wait'] > 100:
        number_of_trips_over_100 += 1

print('Number of trips over 100 mins: ', number_of_trips_over_100)


average_driver_profit = 0
average_pickup_wait_time = 0
average_dropoff_wait_time = 0
for result in T1_results:
    average_driver_profit += result['dropoff_time'] - result['pickup_wait_time']
    average_pickup_wait_time += result['pickup_wait_time']
    average_dropoff_wait_time += result['dropoff_time']


average_driver_profit /= len(T1_results)
average_pickup_wait_time /= len(T1_results)
average_dropoff_wait_time /= len(T1_results)

print('Average driver profit: ', average_driver_profit)
print('Average dropoff wait time: ', average_dropoff_wait_time)
print('Average pickup wait time: ', average_pickup_wait_time)