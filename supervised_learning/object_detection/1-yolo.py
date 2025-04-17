#!/usr/bin/env python3
""" Yolo class """
from tensorflow import keras as K
import numpy as np
import os
import cv2


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
            raise FileNotFoundError(f"Classes file not found: {classes_path}")
        with open(classes_path, 'r') as f:
            class_names = f.read().splitlines()
        return class_names

    def process_outputs(self, outputs, image_size):
        """
        Processes the outputs of the YOLO model.
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Extract the bounding box parameters
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Extract the box confidence and class probabilities
            box_confidence = 1 / (1 + np.exp(-output[..., 4])) 
            class_probs = 1 / (1 + np.exp(-output[..., 5:]))  

            # Generate grid coordinates
            cx = (np.arange(grid_width).reshape(1, -1, 1) + tx) / grid_width
            cy = (np.arange(grid_height).reshape(-1, 1, 1) + ty) / grid_height

            # Calculate the width and height of the bounding boxes
            bw = (np.exp(tw) * self.anchors[i, :, 0]) / image_width
            bh = (np.exp(th) * self.anchors[i, :, 1]) / image_height

            # Convert (cx, cy, bw, bh) to (x1, y1, x2, y2)
            x1 = (cx - bw / 2) * image_width
            y1 = (cy - bh / 2) * image_height
            x2 = (cx + bw / 2) * image_width
            y2 = (cy + bh / 2) * image_height

            # Stack the coordinates into a single array
            box = np.stack([x1, y1, x2, y2], axis=-1)

            # Append processed outputs
            boxes.append(box)
            box_confidences.append(box_confidence[..., np.newaxis])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
