#!/usr/bin/env python3
"""proivdes stats about nginx logs stored in mongoDB"""
from pymongo import MongoClient


if __name__ == '__main__':
    client = MongoClient()
    db = client.logs
    collection = client.logs.nginx

    total_logs = collection.count_documents({})

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    m_count = {method: collection.count_documents({"method": method}) for method in methods}

    # Get the count of status check
    status_check_count = collection.count_documents({"method": "GET", "path": "/status"})

    status_check = collection.count_documents({"method": "GET", "path": "/status"})

    # all stats
    print(f"{total_logs} logs")
    print("Methods:")
    for method, count in m_count.items():
        print(f"\tmethod {method}: {count}")
    print(f"{status_check} status check")
