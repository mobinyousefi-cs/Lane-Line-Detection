#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: tests/test_pipeline.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
Sanity tests for pipeline initialization and a synthetic frame pass.

Usage:
pytest -q

Notes:
- Uses a synthetic lane image to validate fitting path doesn't crash.

=================================================================================================================
"""
from __future__ import annotations

import numpy as np

from lane_ai.pipeline import LaneDetector


def _synthetic_road(h: int = 720, w: int = 1280) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        x_left = int(w * 0.3 + 0.0006 * (y - h) ** 2)
        x_right = int(w * 0.7 - 0.0006 * (y - h) ** 2)
        img[y, max(0, x_left - 2) : x_left + 2] = (255, 255, 255)
        img[y, max(0, x_right - 2) : x_right + 2] = (255, 255, 255)
    return img


def test_pipeline_runs_on_synthetic():
    cfg = {
        "s_thresh": (100, 255),
        "l_thresh": (150, 255),
        "roi": {"mode": "auto"},
    }
    det = LaneDetector(cfg)
    frame = _synthetic_road()
    out = det.process_frame(frame)
    assert out.shape == frame.shape