#!/usr/bin/env python3
""" Yolo class """
import numpy as np
from tensorflow import keras as K


class Yolo:
    """YOLO (You Only Look Once) object detection model class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        initialize the YOLO model.
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self._load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def _load_classes(self, classes_path):
        """Load class names from file"""
        with open(classes_path, 'r') as f:
            return f.read().splitlines()

    @staticmethod
    def _sigmoid(x):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        process Darknet model outputs to obtain bounding boxes, confidences,
        and class probabilities.
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        # iterate through each output layer
        for i, output in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, _ = output.shape

            # get box parameters
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # get grid indices
            cx = np.arange(grid_w).reshape(1, grid_w, 1)
            cy = np.arange(grid_h).reshape(grid_h, 1, 1)

            # get box centers
            bx = (self._sigmoid(tx) + cx) / grid_w
            by = (self._sigmoid(ty) + cy) / grid_h

            # get box dimensions
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            # normalize by width
            bw = (np.exp(tw) * pw) / self.model.input.shape[2]
            # normalize by height
            bh = (np.exp(th) * ph) / self.model.input.shape[1]

            # convert to absolute coordinates
            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            # stack coordinates
            box_coords = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box_coords)

            # process confidences and class probabilities
            box_confidences.append(self._sigmoid(output[..., 4:5]))
            box_class_probs.append(self._sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs
