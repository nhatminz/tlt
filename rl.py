#!/usr/bin/env python3
"""Compatibility entry point for the FastGRPO training program."""
import runpy
from pathlib import Path
if __name__=='__main__':runpy.run_path(str(Path(__file__).with_name('grpo_speculative.py')),run_name='__main__')
