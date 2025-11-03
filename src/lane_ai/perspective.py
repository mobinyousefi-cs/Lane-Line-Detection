#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: perspective.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
ROI estimation and perspective (bird's‑eye) transform utilities.

Usage:
from lane_ai.perspective import auto_roi, get_perspective_matrices

Notes:
- ROI points are normalized (0..1), making it resolution‑agnostic.

=================================================================================================================
"""
from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple


def denorm_points(points, w: int, h: int):
    return np.float32([[x * w, y * h] for x, y in points])


def auto_roi(binary: np.ndarray) -> np.ndarray:
    """Compute a trapezoid ROI automatically from binary lane mask."""
    h, w = binary.shape[:2]
    y_bottom = int(h * 0.95)
    y_top = int(h * 0.6)
    margin_bottom = int(w * 0.15)
    margin_top = int(w * 0.4)
    src = np.float32([
        [margin_bottom, y_bottom], [w - margin_bottom, y_bottom], [w - margin_top, y_top], [margin_top, y_top]
    ])
    return src


def get_perspective_matrices(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    return M, Minv


def warp(img: np.ndarray, M: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)