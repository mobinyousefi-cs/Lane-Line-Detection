#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: thresholds.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
Color & gradient threshold utilities.

Usage:
from lane_ai.thresholds import combined_threshold

Notes:
- Works best on daylight road scenes; tune thresholds in config.yaml.

=================================================================================================================
"""
from __future__ import annotations

import cv2
import numpy as np


def to_colorspace(img: np.ndarray, space: str) -> np.ndarray:
    if space.upper() == "HLS":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    if space.upper() == "YUV":
        return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    if space.upper() == "LAB":
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return img


def sobel_binary(gray: np.ndarray, orient: str = "x", ksize: int = 3, mag_thresh=(30, 255)) -> np.ndarray:
    dx = 1 if orient == "x" else 0
    dy = 0 if orient == "x" else 1
    sob = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
    abs_sob = np.absolute(sob)
    scaled = np.uint8(255 * abs_sob / np.max(abs_sob + 1e-6))
    binary = np.zeros_like(scaled)
    binary[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 255
    return binary


def combined_threshold(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Return a binary mask (uint8 0/255) highlighting lane pixels."""
    space = cfg.get("color_space", "HLS").upper()
    s_min, s_max = cfg.get("s_thresh", [120, 255])
    l_min, l_max = cfg.get("l_thresh", [200, 255])

    cimg = to_colorspace(img, space)

    if space == "HLS":
        h, l, s = cv2.split(cimg)
        s_bin = cv2.inRange(s, s_min, s_max)
        l_bin = cv2.inRange(l, l_min, l_max)
        color_bin = cv2.bitwise_and(s_bin, l_bin)
    elif space == "YUV":
        y, u, v = cv2.split(cimg)
        color_bin = cv2.inRange(y, l_min, 255)
    else:
        # Fallback: simple grayscale threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, color_bin = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    if cfg.get("sobel", {}).get("enable", True):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        s_cfg = cfg.get("sobel", {})
        sob = sobel_binary(gray, s_cfg.get("orient", "x"), s_cfg.get("ksize", 3), s_cfg.get("mag_thresh", (30, 255)))
        combined = cv2.bitwise_or(color_bin, sob)
    else:
        combined = color_bin

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    return combined