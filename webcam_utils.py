import cv2
import numpy as np
import torch
import os
import sys


def preprocess_for_snn(frame, roi=None):
    if roi:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(
            frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    clahe    = cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(
        enhanced, (3, 3), 0)
    thresh   = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    kernel  = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, kernel)
    resized = cv2.resize(
        cleaned, (28, 28),
        interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class DVSSimulator:
    def __init__(self, threshold=0.03):
        self.threshold  = threshold
        self.prev_frame = None

    def process(self, frame_gray):
        if self.prev_frame is None:
            self.prev_frame = frame_gray.copy()
            return (np.zeros_like(frame_gray),
                    np.zeros_like(frame_gray), 0)
        diff = (frame_gray.astype(float) -
                self.prev_frame.astype(float))
        on_events  = (
            diff >  self.threshold).astype(float)
        off_events = (
            diff < -self.threshold).astype(float)
        n_events   = int(
            on_events.sum() + off_events.sum())
        self.prev_frame = frame_gray.copy()
        return on_events, off_events, n_events

    def reset(self):
        self.prev_frame = None
