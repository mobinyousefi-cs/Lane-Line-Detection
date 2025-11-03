#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: __init__.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
Package initializer for lane_ai. Exposes key APIs for programmatic use.

Usage:
from lane_ai.pipeline import LaneDetector

Notes:
- Keep exports minimal to maintain a clean namespace.

=================================================================================================================
"""
from .pipeline import LaneDetector

__all__ = ["LaneDetector"]