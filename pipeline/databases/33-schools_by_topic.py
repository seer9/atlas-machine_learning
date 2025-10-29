#!/usr/bin/env python3
""" lists all schools with a specific topic """


def schools_by_topic(mongo_collection, topic):
    """lists all schools with a specific topic.

    Args:
        mongo_collection: pymongo collection object
        topic: topic searched

    Returns: list of schools
    """
    list = mongo_collection.find({'topics': topic})
    return list
