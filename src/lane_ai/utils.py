#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: utils.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
Utility helpers: polynomial fit, curvature, vehicle offset, smoothing.

Usage:
from lane_ai.utils import fit_polynomial, measure_curvature, lane_offset

Notes:
- Pixel-to-meter conversion assumes typical lane width; tune as needed.

=================================================================================================================
"""
from __future__ import annotations

import collections
from typing import Deque, Tuple

import cv2
import numpy as np


def fit_polynomial(binary_warped: np.ndarray, nwindows: int = 9, margin: int = 100, minpix: int = 50):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2 :, :], axis=0)
    out_img = np.dstack([binary_warped, binary_warped, binary_warped])

    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = binary_warped.shape[0] // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        )
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        )

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if good_left_inds.any():
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if good_right_inds.any():
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
    right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

    return left_fit, right_fit


def measure_curvature(binary_warped: np.ndarray, left_fit, right_fit) -> Tuple[float, float]:
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    def calc_curvature(fit):
        if fit is None:
            return float("nan")
        y_eval = np.max(ploty)
        # Convert to meters
        fit_cr = [fit[0] * xm_per_pix / (ym_per_pix ** 2), fit[1] * xm_per_pix / ym_per_pix, fit[2] * xm_per_pix]
        A, B = fit_cr[0], fit_cr[1]
        return ((1 + (2 * A * y_eval * ym_per_pix + B) ** 2) ** 1.5) / abs(2 * A + 1e-6)

    return calc_curvature(left_fit), calc_curvature(right_fit)


def lane_offset(binary_warped: np.ndarray, left_fit, right_fit) -> float:
    xm_per_pix = 3.7 / 700
    h, w = binary_warped.shape[:2]
    y = h - 1
    if left_fit is None or right_fit is None:
        return float("nan")
    left_x = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
    right_x = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
    lane_center = (left_x + right_x) / 2
    vehicle_center = w / 2
    offset_pixels = vehicle_center - lane_center
    return offset_pixels * xm_per_pix


class RollingAverage:
    def __init__(self, maxlen: int = 5):
        self.left: Deque[np.ndarray] = collections.deque(maxlen=maxlen)
        self.right: Deque[np.ndarray] = collections.deque(maxlen=maxlen)

    def update(self, left_fit, right_fit):
        if left_fit is not None:
            self.left.append(left_fit)
        if right_fit is not None:
            self.right.append(right_fit)

    @property
    def avg(self):
        l = np.mean(self.left, axis=0) if len(self.left) else None
        r = np.mean(self.right, axis=0) if len(self.right) else None
        return l, r