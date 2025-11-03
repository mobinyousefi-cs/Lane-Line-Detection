#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================================
Project: Lane Line Detection using AI
File: models/unet.py
Author: Mobin Yousefi (GitHub: github.com/mobinyousefi-cs)
Created: 2025-11-03
Updated: 2025-11-03
License: MIT License (see LICENSE file for details)
=

Description:
Lightweight U-Net skeleton (no training loop) for future lane segmentation work.

Usage:
# Placeholder for future DL experiments; not used in classical pipeline.

Notes:
- Keep as a minimal stub to avoid heavy dependencies in the base install.

=================================================================================================================
"""
from __future__ import annotations

# Intentionally minimal to keep base dependencies light. Replace with a real model later.
try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = object  # type: ignore


class TinyUNet(nn.Module):  # type: ignore[misc]
    def __init__(self, in_ch=3, out_ch=1, feat=32):
        super().__init__()
        if torch is None:
            raise ImportError("PyTorch is not installed. Install it to use TinyUNet.")

        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_ch, feat)
        self.enc2 = conv_block(feat, feat * 2)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = conv_block(feat * 2 + feat, feat)
        self.final = nn.Conv2d(feat, out_ch, 1)

    def forward(self, x):  # type: ignore[override]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        up = torch.nn.functional.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up, e1], dim=1))
        return self.final(d1)