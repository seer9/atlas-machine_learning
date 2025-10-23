#!/usr/bin/env python3
"""first launch api task"""
import requests


if __name__ == "__main__":
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    response = requests.get(url)
    data = response.json()

    # Sort launches by date_unix
    sorted_launches = sorted(data, key=lambda x: x['date_unix'])

    # Get the first launch
    first_launch = sorted_launches[0]

    # Extract required information
    launch_name = first_launch['name']
    date_local = first_launch['date_local']
    rocket_id = first_launch['rocket']
    launchpad_id = first_launch['launchpad']

    # Fetch rocket details
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket_id}"
    rocket_response = requests.get(rocket_url)
    rocket_name = rocket_response.json()['name']

    # Fetch launchpad details
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad_id}"
    launchpad_response = requests.get(launchpad_url)
    launchpad_data = launchpad_response.json()
    lp_name = launchpad_data['name']
    lp_local = launchpad_data['locality']

    # Display the formatted result
    print(
        f"{launch_name} ({date_local}) {rocket_name} - {lp_name} ({lp_local})")
