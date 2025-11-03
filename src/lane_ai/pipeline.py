#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: pipeline.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
End-to-end lane detection pipeline: thresholding, perspective transform, sliding window search,
polynomial fit, curvature/offset metrics, and overlay rendering.

Usage:
from lane_ai.pipeline import LaneDetector

Notes:
- Designed for 720p–1080p videos; configurable via config dict.

=================================================================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .thresholds import combined_threshold
from .perspective import auto_roi, denorm_points, get_perspective_matrices, warp
from .utils import RollingAverage, fit_polynomial, lane_offset, measure_curvature


@dataclass
class PipelineConfig:
    color_space: str = "HLS"
    s_thresh: Tuple[int, int] = (120, 255)
    l_thresh: Tuple[int, int] = (200, 255)
    sobel: dict = None
    roi: dict = None
    perspective: dict = None
    sliding_window: dict = None
    smoothing_frames: int = 5


class LaneDetector:
    def __init__(self, cfg: dict):
        self.cfg = self._merge_defaults(cfg)
        self.smoother = RollingAverage(self.cfg["smoothing_frames"])
        self._M = None
        self._Minv = None

    @staticmethod
    def _merge_defaults(cfg: dict) -> dict:
        default = {
            "color_space": "HLS",
            "s_thresh": (120, 255),
            "l_thresh": (200, 255),
            "sobel": {"enable": True, "orient": "x", "ksize": 3, "mag_thresh": (30, 255)},
            "roi": {"mode": "auto", "manual_points": [[0.15, 0.95], [0.85, 0.95], [0.6, 0.6], [0.4, 0.6]]},
            "perspective": {"dst_points": [[0.2, 1.0], [0.8, 1.0], [0.8, 0.0], [0.2, 0.0]]},
            "sliding_window": {"nwindows": 9, "margin": 100, "minpix": 50},
            "smoothing_frames": 5,
        }
        default.update(cfg or {})
        return default

    def _prepare_perspective(self, binary: np.ndarray) -> None:
        h, w = binary.shape[:2]
        roi_cfg = self.cfg["roi"]
        if roi_cfg.get("mode", "auto") == "auto":
            src = auto_roi(binary)
        else:
            src = denorm_points(roi_cfg.get("manual_points"), w, h)
        dst = denorm_points(self.cfg["perspective"]["dst_points"], w, h)
        self._M, self._Minv = get_perspective_matrices(src, dst)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # 1) Thresholding
        binary = combined_threshold(frame, self.cfg)

        # 2) Perspective matrices (lazy init per resolution)
        if self._M is None or self._Minv is None:
            self._prepare_perspective(binary)

        # 3) Warp to bird's-eye
        h, w = binary.shape[:2]
        warped = warp(binary, self._M, (w, h))

        # 4) Sliding window fit
        sw = self.cfg["sliding_window"]
        left_fit, right_fit = fit_polynomial(warped, sw["nwindows"], sw["margin"], sw["minpix"])

        # 5) Smoothing
        self.smoother.update(left_fit, right_fit)
        left_fit, right_fit = self.smoother.avg

        # 6) Curvature & offset
        left_curv, right_curv = measure_curvature(warped, left_fit, right_fit)
        offset_m = lane_offset(warped, left_fit, right_fit)

        # 7) Render overlay
        overlay = self._render_lane(frame, warped, left_fit, right_fit)
        self._annotate(overlay, left_curv, right_curv, offset_m)
        return overlay

    def _render_lane(self, frame: np.ndarray, warped_bin: np.ndarray, left_fit, right_fit) -> np.ndarray:
        h, w = warped_bin.shape[:2]
        ploty = np.linspace(0, h - 1, h)
        color_warp = np.zeros((h, w, 3), dtype=np.uint8)

        if left_fit is not None and right_fit is not None:
            left_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            pts_left = np.array([np.transpose(np.vstack([left_x, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, ploty])))])
            pts = np.hstack((pts_left, pts_right)).astype(np.int32)
            cv2.fillPoly(color_warp, pts, (0, 255, 0))

        newwarp = cv2.warpPerspective(color_warp, self._Minv, (frame.shape[1], frame.shape[0]))
        result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
        return result

    @staticmethod
    def _annotate(img: np.ndarray, left_curv: float, right_curv: float, offset_m: float) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text1 = f"Curvature: L={left_curv:6.0f} m  R={right_curv:6.0f} m"
        text2 = f"Vehicle offset: {offset_m:+.2f} m ( + = right )"
        cv2.putText(img, text1, (40, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text2, (40, 100), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)