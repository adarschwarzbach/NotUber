import json

#  T1_results.json | T1_extra.txt | T2_results.json
with open('T2_results.json', 'r') as f:
    T1_results = json.load(f)

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


