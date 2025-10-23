#!/usr/bin/env python3
"""rocket frequency api task"""
import requests


if __name__ == "__main__":
    rocket_launch_count = {}
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    # Fetch launches data
    launches = requests.get(launches_url).json()

    # Count launches per rocket
    for launch in launches:
        rocket_id = launch['rocket']
        if rocket_id in rocket_launch_count:
            rocket_launch_count[rocket_id] += 1
        else:
            rocket_launch_count[rocket_id] = 1

    # Fetch rocket names
    rockets = requests.get(rockets_url).json()
    rocket_names = {rocket['id']: rocket['name'] for rocket in rockets}

    # Map rocket IDs to names and sort results
    result = [
        (rocket_names[rocket_id], count)
        for rocket_id, count in rocket_launch_count.items()
        if rocket_id in rocket_names
    ]
    result.sort(key=lambda x: (-x[1], x[0]))

    # Print results
    for rocket_name, count in result:
        print(f"{rocket_name}: {count}")
    