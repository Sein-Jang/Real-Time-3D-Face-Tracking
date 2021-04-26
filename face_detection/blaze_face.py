"""
* Author
Sein Jang

* Reference
ibaiGorordo / BlazeFace-TFLite-Inference
https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference
"""

import cv2
import numpy as np
import tensorflow as tf

from .utils import gen_anchors, SsdAnchorsCalculatorOptions

KEY_POINT_SIZE = 6
MAX_FACE_NUM = 50


class FaceDetector:
    def __init__(self, model_type="front", score_threshold=0.8, iou_threshold=0.3):
        self.image_height, self.image_width, self.image_channels = None, None, None
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.interpreter = self.initialize_model(model_type)
        self.anchors = self.generate_anchors(model_type)

        self.input_details, self.input_height, self.input_width, self.input_channels = self.get_input_details()
        self.output_details = self.get_output_details()

    @staticmethod
    def initialize_model(model_type):
        if model_type == "front":
            interpreter = tf.lite.Interpreter(model_path="face_detection/model/face_detection_front.tflite")
        elif model_type == "back":
            interpreter = tf.lite.Interpreter(model_path="face_detection/model/face_detection_back.tflite")
        else:
            raise Exception("The model type must be either 'front' or 'back'.")
        interpreter.allocate_tensors()

        return interpreter

    @staticmethod
    def generate_anchors(model_type):
        if model_type == "front":
            # Options to generate anchors for SSD object detection models.
            ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=128, input_size_height=128,
                                                                         min_scale=0.1484375, max_scale=0.75,
                                                                         anchor_offset_x=0.5, anchor_offset_y=0.5,
                                                                         num_layers=4,
                                                                         feature_map_width=[], feature_map_height=[],
                                                                         strides=[8, 16, 16, 16], aspect_ratios=[1.0],
                                                                         reduce_boxes_in_lowest_layer=False,
                                                                         interpolated_scale_aspect_ratio=1.0,
                                                                         fixed_anchor_size=True)

        elif model_type == "back":
            # Options to generate anchors for SSD object detection models.
            ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(input_size_width=256, input_size_height=256,
                                                                         min_scale=0.15625, max_scale=0.75,
                                                                         anchor_offset_x=0.5, anchor_offset_y=0.5,
                                                                         num_layers=4,
                                                                         feature_map_width=[], feature_map_height=[],
                                                                         strides=[16, 32, 32, 32], aspect_ratios=[1.0],
                                                                         reduce_boxes_in_lowest_layer=False,
                                                                         interpolated_scale_aspect_ratio=1.0,
                                                                         fixed_anchor_size=True)

        else:
            raise Exception("The model type must be either 'front' or 'back'.")

        return gen_anchors(ssd_anchors_calculator_options)

    def detect_faces(self, image):
        # input image for inference
        input_tensor = self.prepare_input_image(image)

        # inference
        faces, scores = self.inference(input_tensor)

        # get details (boxes, keypoints)
        detection_results = self.get_detection_results(faces, scores)

        return detection_results

    def get_input_details(self):
        input_details = self.interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_height, input_width, input_channels = input_shape[1], input_shape[2], input_shape[3]

        return input_details, input_height, input_width, input_channels

    def get_output_details(self):
        output_details = self.interpreter.get_output_details()

        return output_details

    def prepare_input_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width, self.image_channels = image.shape

        # Input = (-1, 1) with 128x128(front) / 256x256(back)
        image = image / 255.0
        image_resized = tf.image.resize(image, [self.input_height, self.input_width],
                                        method='bicubic', preserve_aspect_ratio=False)
        image_input = image_resized.numpy()
        image_input = (image_input - 0.5) / 0.5

        reshape_image = image_input.reshape(1, self.input_height, self.input_width, self.input_channels)
        input_tensor = tf.convert_to_tensor(reshape_image, dtype=tf.float32)

        return input_tensor

    def get_detection_results(self, faces, scores):
        detections = np.where(scores > self.score_threshold)[0]
        scores = np.exp(-scores[detections])

        num_detections = detections.shape[0]

        keypoints = np.zeros((num_detections, KEY_POINT_SIZE, 2))
        boxes = np.zeros((num_detections, 4))

        for idx, detection_idx in enumerate(detections):
            anchor = self.anchors[detection_idx]

            sx = faces[detection_idx, 0]
            sy = faces[detection_idx, 1]
            w = faces[detection_idx, 2]
            h = faces[detection_idx, 3]

            cx = sx + anchor.x_center * self.input_width
            cy = sy + anchor.y_center * self.input_height

            cx /= self.input_width
            cy /= self.input_height
            w /= self.input_width
            h /= self.input_height

            for j in range(KEY_POINT_SIZE):
                lx = faces[detection_idx, 4 + (2 * j) + 0]
                ly = faces[detection_idx, 4 + (2 * j) + 1]
                lx += anchor.x_center * self.input_width
                ly += anchor.y_center * self.input_height
                lx /= self.input_width
                ly /= self.input_height
                keypoints[idx, j, :] = np.array([lx, ly])

            boxes[idx, :] = np.array([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5])

        # Filter based on non max suppression
        filtered_detection_indices = tf.image.non_max_suppression(boxes, scores, MAX_FACE_NUM, self.iou_threshold)

        filtered_boxes = tf.gather(boxes, filtered_detection_indices).numpy()
        filtered_keypoints = tf.gather(keypoints, filtered_detection_indices).numpy()
        filtered_scores = tf.gather(scores, filtered_detection_indices).numpy()

        detection_results = dict(boxes=filtered_boxes,
                                 keypoints=filtered_keypoints,
                                 scores=filtered_scores)

        return detection_results

    def inference(self, input_tensor):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()

        # 896x16 matrix (detected faces) / matrix (detection scores)
        faces = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))
        scores = np.squeeze(self.interpreter.get_tensor(self.output_details[1]['index']))

        return faces, scores

    def draw_detections(self, img, results):
        bounding_boxes = results['boxes']
        keypoints = results['keypoints']
        scores = results['scores']

        # Add bounding boxes and keypoints
        for bounding_box, keypoints, score in zip(bounding_boxes, keypoints, scores):
            x_1 = (self.image_width * bounding_box[0]).astype(int)
            x_2 = (self.image_width * bounding_box[2]).astype(int)
            y_1 = (self.image_height * bounding_box[1]).astype(int)
            y_2 = (self.image_height * bounding_box[3]).astype(int)
            cv2.rectangle(img, (x_1, y_1), (x_2, y_2), (22, 22, 250), 2)
            cv2.putText(img, '{:.2f}'.format(score), (x_1, y_1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (22, 22, 250), 2)

            # Add keypoints for the current face
            for keypoint in keypoints:
                _x = (keypoint[0] * self.image_width).astype(int)
                _y = (keypoint[1] * self.image_height).astype(int)
                cv2.circle(img, (_x, _y), 4, (214, 202, 18), -1)

        return img
