#!/usr/bin/env python3
"""lists all docs in a collection"""


def list_all(mongo_collection):
    """Lists all documents in a collection."""
    return list(mongo_collection.find())
