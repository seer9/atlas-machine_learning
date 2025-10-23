#!/usr/bin/env python3
"""listing home planets of sentient species"""
import requests


def sentientPlanets():
    """
    Get home planets of sentient species from swapi api.
    Returns:
        planets: list of home planets of sentient species
    """
    url = "https://swapi.dev/api/species/"
    planets = []
    while url:
        response = requests.get(url, headers={"Accept": "application/json"},
                                params={"term": "sentient"})
        data = response.json()

        for species in data['results']:
            if species['classification'] == 'sentient' or \
                    species['designation'] == 'sentient':
                if species['homeworld'] is not None:
                    homeworld = requests.get(species['homeworld'])
                    planets.append(homeworld.json()['name'])

        url = response.json()['next']
    return planets
