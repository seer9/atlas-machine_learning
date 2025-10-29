#!/usr/bin/env python3
""" inserts a new document in a collection """


def insert_school(mongo_collection, **kwargs):
    """Inserts a new document in a collection.

    Args:
        mongo_collection: pymongo collection object
        **kwargs: key/value pairs of the new document

    Returns:
        The new id
    """
    newid = mongo_collection.insert_one(kwargs).inserted_id
    return newid
