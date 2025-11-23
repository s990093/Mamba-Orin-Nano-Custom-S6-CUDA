#!/usr/bin/env python3
"""
Quick fix test - don't pre-convert A_log to A, let Metal kernel handle it
"""

Note: The issue is that we're pre-converting A_log to A in Python:
  block.A = -np.exp(a_log)  # Line 201 in run_inference.py

But the Metal kernel expects A_log and does:
  float dA = exp(-exp(A_val) * dt_val);  # Line 226 in mamba_ssm_full.metal

This means:
1. Python: A = -exp(A_log)
2. Metal: dA = exp(-exp(A) * dt) = exp(-exp(-exp(A_log)) * dt) = WRONG!

The Metal kernel should receive A_log, not A.

Solution: Don't convert A_log in Python, keep it as A_log and pass it to Metal.
