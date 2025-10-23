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
            location = data.get("location")
            print(location)
        elif response.status_code == 404:
            print("Not found")
        elif response.status_code == 403:
            reset = int(
                response.headers.get("X-RateLimit-Reset", time.time()))
            elapsed = (reset - int(time.time())) // 60
            print(f"Reset in {elapsed} min")
    else:
        print("Usage: ./2-user_location.py <GitHub API URL>")
