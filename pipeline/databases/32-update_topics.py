#!/usr/bin/env python3
""" updates all topics of a school document """


def update_topics(mongo_collection, name, topics):
    """updates all topics of a school document.

    Args:
        mongo_collection: pymongo collection object
        name: name of the school to update
        topics: list of topics approached in the school

    Returns:
        None
    """
    mongo_collection.update_many(
        {'name': name},
        {'$set': {'topics': topics}}
    )
