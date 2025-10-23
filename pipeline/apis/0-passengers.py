#!/usr/bin/env python3
"""swapi api task"""
import requests


def availableShips(passengerCount):
    """
    Get available ships from swapi api.
    Args:
        passengerCount: number of passengers
    Returns:
        ships: list of available ships
    """
    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        response = requests.get(url)
        data = response.json()

        for ship in data['results']:
            try:
                max_passengers = int(ship['passengers'].replace(',', ''))
                if max_passengers >= passengerCount:
                    ships.append(ship['name'])
            except ValueError:
                continue

        url = data['next']

    return ships