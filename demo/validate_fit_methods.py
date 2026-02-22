#!/usr/bin/env python3
"""
One-off validation: confirm that the standalone demo implementations of
fit_approach2, fit_approach3, and fit_vpnls produce identical exponent
estimates (to machine precision) as the codebase implementations.

Usage: python demo/validate_fit_methods.py
"""

import numpy as np

# --- Demo implementations (standalone) ---
from demo.chinchilla_fit_methods import (  # type: ignore
    Data,
    generate_data,
    fit_approach2 as demo_approach2,
    fit_approach3 as demo_approach3,
    fit_vpnls as demo_vpnls,
)

# --- Codebase implementations ---
from scaling_law_analysis.chinchilla import (
    fit_approach2 as lib_approach2,
    fit_approach3 as lib_approach3,
    fit_vpnls as lib_vpnls,
)


def main() -> None:
    # Generate a single dataset under benign conditions (low noise, wide grid,
    # many points) so Approach 2 parabola fits never fail.
    rng = np.random.default_rng(12345)
    budgets = np.logspace(17, 21, 5)
    data = generate_data(
        budgets=budgets,
        n_points=31,
        log_width=np.log10(8),
        noise_std=0.02,
        rng=rng,
    )

    print("=" * 70)
    print("Validation: demo vs codebase fitting implementations")
    print("=" * 70)
    print(f"  Data: {len(data.N)} points, 5 budgets, 31 pts/curve, ±8× width, σ=0.02")
    print()

    all_pass = True

    # --- Approach 2 ---
    demo_a2 = demo_approach2(data)
    lib_r2 = lib_approach2(N=data.N, D=data.D, L=data.L, C=data.C)
    lib_a2 = (lib_r2.a, lib_r2.b)

    print("Approach 2:")
    print(f"  Demo:     a={demo_a2[0]:.15f}, b={demo_a2[1]:.15f}")
    print(f"  Codebase: a={lib_a2[0]:.15f}, b={lib_a2[1]:.15f}")
    a2_a_diff = abs(demo_a2[0] - lib_a2[0])
    a2_b_diff = abs(demo_a2[1] - lib_a2[1])
    a2_ok = a2_a_diff < 1e-12 and a2_b_diff < 1e-12
    print(
        f"  Δa={a2_a_diff:.2e}, Δb={a2_b_diff:.2e}  {'✓ PASS' if a2_ok else '✗ FAIL'}"
    )
    if not a2_ok:
        all_pass = False
    print()

    # --- Approach 3 ---
    demo_a3 = demo_approach3(data)
    lib_r3 = lib_approach3(N=data.N, D=data.D, L=data.L)
    lib_a3 = (lib_r3.a, lib_r3.b)

    print("Approach 3:")
    print(f"  Demo:     a={demo_a3[0]:.15f}, b={demo_a3[1]:.15f}")
    print(f"  Codebase: a={lib_a3[0]:.15f}, b={lib_a3[1]:.15f}")
    a3_a_diff = abs(demo_a3[0] - lib_a3[0])
    a3_b_diff = abs(demo_a3[1] - lib_a3[1])
    a3_ok = a3_a_diff < 1e-12 and a3_b_diff < 1e-12
    print(
        f"  Δa={a3_a_diff:.2e}, Δb={a3_b_diff:.2e}  {'✓ PASS' if a3_ok else '✗ FAIL'}"
    )
    if not a3_ok:
        all_pass = False
    print()

    # --- VPNLS ---
    demo_vp = demo_vpnls(data)
    lib_rv = lib_vpnls(N=data.N, D=data.D, L=data.L)
    lib_vp = (lib_rv.a, lib_rv.b)

    print("VPNLS:")
    print(f"  Demo:     a={demo_vp[0]:.15f}, b={demo_vp[1]:.15f}")
    print(f"  Codebase: a={lib_vp[0]:.15f}, b={lib_vp[1]:.15f}")
    vp_a_diff = abs(demo_vp[0] - lib_vp[0])
    vp_b_diff = abs(demo_vp[1] - lib_vp[1])
    vp_ok = vp_a_diff < 1e-12 and vp_b_diff < 1e-12
    print(
        f"  Δa={vp_a_diff:.2e}, Δb={vp_b_diff:.2e}  {'✓ PASS' if vp_ok else '✗ FAIL'}"
    )
    if not vp_ok:
        all_pass = False
    print()

    print("=" * 70)
    if all_pass:
        print("ALL METHODS MATCH TO MACHINE PRECISION ✓")
    else:
        print("SOME METHODS DIFFER — investigate discrepancies above")
    print("=" * 70)


if __name__ == "__main__":
    main()
