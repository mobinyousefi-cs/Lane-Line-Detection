#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: main.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
Command-line interface to run lane detection on images or videos.

Usage:
python -m lane_ai.main --image path/to.jpg --show
python -m lane_ai.main --video path/to.mp4 --out outputs/out.mp4 --display

Notes:
- Uses config.yaml for thresholds and ROI. Override via CLI flags as needed.

=================================================================================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import yaml

from .pipeline import LaneDetector
from .video import VideoProcessor


def load_config(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lane Line Detection using AI (classical CV)")
    p.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=Path, help="Path to an image")
    g.add_argument("--video", type=Path, help="Path to a video file")
    p.add_argument("--out", type=Path, default=Path("outputs/out.mp4"), help="Output video path")
    p.add_argument("--display", action="store_true", help="Display video while processing")
    p.add_argument("--show", action="store_true", help="Show image output in a window")
    return p.parse_args(argv)


def run_image(detector: LaneDetector, image_path: Path, show: bool) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    out = detector.process_frame(img)
    out_path = Path("outputs") / (image_path.stem + "_lanes.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    if show:
        cv2.imshow("Lane Detection", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(detector: LaneDetector, video_path: Path, out_path: Path, display: bool) -> None:
    vp = VideoProcessor(detector)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vp.process_video(str(video_path), str(out_path), display=display)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    detector = LaneDetector(cfg.get("pipeline", {}))

    if args.image:
        run_image(detector, args.image, args.show)
    else:
        run_video(detector, args.video, args.out, args.display)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))