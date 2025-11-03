#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: video.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
Video processing utilities to stream frames through the LaneDetector.

Usage:
from lane_ai.video import VideoProcessor

Notes:
- Uses OpenCV VideoCapture/VideoWriter. Set --display to view live.

=================================================================================================================
"""
from __future__ import annotations

import cv2


class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector

    def process_video(self, in_path: str, out_path: str, display: bool = False) -> None:
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {in_path}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed = self.detector.process_frame(frame)
            out.write(processed)
            if display:
                cv2.imshow("Lane Detection", processed)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()