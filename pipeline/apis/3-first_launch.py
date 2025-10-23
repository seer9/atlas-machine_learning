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
    name = first_launch['name']
    date = first_launch['date']
    rocket = first_launch['rocket']
    launchpad = first_launch['launchpad']

    # Fetch rocket details
    rocket_url = f"https://api.spacexdata.com/v4/rockets/{rocket}"
    rocket_response = requests.get(rocket_url)
    r_name = rocket_response.json()['name']

    # Fetch launchpad details
    launchpad_url = f"https://api.spacexdata.com/v4/launchpads/{launchpad}"
    launchpad_response = requests.get(launchpad_url)
    launchpad_data = launchpad_response.json()
    launchpad_name = launchpad_data['name']
    launchpad_locality = launchpad_data['locality']

    # Display the formatted result
    print(
        f"{name} ({date}) {r_name} - {launchpad_name} ({launchpad_locality})")
