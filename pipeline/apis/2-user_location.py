#!/usr/bin/env python3
"""user location api task"""
import requests
import sys
import time


if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            location = data.get("location", "Location not available")
            print(location)
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time()))
            minutes_to_reset = (reset_time - int(time.time())) // 60
            print(f"Reset in {minutes_to_reset} min")
    else:
        print("Usage: ./2-user_location.py <GitHub API URL>")
