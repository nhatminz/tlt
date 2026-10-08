#!/usr/bin/env python3
"""TLT-native entry point for the source FastGRPO full-vocabulary tuner."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tune_opd_proposals import *
if __name__=='__main__':main()
