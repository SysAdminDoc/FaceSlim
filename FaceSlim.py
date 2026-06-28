#!/usr/bin/env python3
"""Compatibility launcher for FaceSlim v1.14.0."""

import multiprocessing

multiprocessing.freeze_support()

from FaceSlim_v1 import main


if __name__ == "__main__":
    main()
