#!/usr/bin/env python3
""" Yolo class """
from tensorflow import keras as K
import numpy as np


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
        with open(classes_path, 'r') as f:
            class_names = f.read().splitlines()
        return class_names
    
    def sigmoid(self, x):
        """ Computes the sigmoid activation function """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the YOLO model.
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes = output.shape[:3]

            tx = self.sigmoid(output[..., 0])
            ty = self.sigmoid(output[..., 1])
            tw = output[..., 2]
            th = output[..., 3]

            # Generate grid coordinates
            cx, cy = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            # Calculate bx, by, bw, bh
            bx = (tx + cx) / grid_w
            by = (ty + cy) / grid_h
            bw = np.exp(tw) * self.anchors[i, :, 0] / image_width
            bh = np.exp(th) * self.anchors[i, :, 1] / image_height

            # Convert to corner coordinates
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            # Ensure the coordinates are relative to the image size
            x1 = x1 - (image_width / 2)
            y1 = y1 - (image_height / 2)
            x2 = x2 - (image_width / 2)
            y2 = y2 - (image_height / 2)

            # Append processed boxes
            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))

            # Extract and append box confidences
            box_confidences.append(self.sigmoid(output[..., 4:5]))

            # Extract and append class probabilities
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs
