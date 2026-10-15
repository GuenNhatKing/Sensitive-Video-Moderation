#!/usr/bin/env python3
"""
Wrapper tương thích ngược cho resnet_adult_video_detect.py.
Chuyển tiếp trực tiếp sang proposed_adult_video_detect.py (Pipeline 4 giai đoạn đề xuất).
"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from proposed.proposed_adult_video_detect import main

if __name__ == "__main__":
    main()