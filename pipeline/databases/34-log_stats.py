#!/usr/bin/env python3
"""proivdes stats about nginx logs stored in mongoDB"""
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient()
    db = client.logs
    collection = client.logs.nginx

    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")

    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    status_check = collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_check} status check")