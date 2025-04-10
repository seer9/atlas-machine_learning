#!/usr/bin/env python3
""" Yolo class """
from tensorflow import keras as K
import os


class Yolo:
    """ Yolo class """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ constructor """
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_classes(self, classes_path):
        """ loads the class names from a file """
        if not os.path.isfile(classes_path):
            raise None
        with open(classes_path, 'r') as f:
            class_names = f.read().splitlines()
        return class_names
