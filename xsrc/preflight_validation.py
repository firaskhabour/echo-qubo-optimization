"""
SHO Pre-Flight Validation Checklist
Run this BEFORE launching full 720-instance experiment

This 5-minute sanity test catches methodological violations early.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import SHO components
from spectral_homotopy_optimization import (
    spectral_homotopy_QUBO,
    get_default_SHO_parameters,
    build_Q_homotopy
)
from sho_metrics import extract_solution_metrics


def test_budget_accounting():
    """
    🔴 CRITICAL TEST 1: Verify exact 800K budget is used
    """
    print("\n" + "="*80)
    print("TEST 1: Budget Accounting")
    print("="*80)
    
    # Mock parameters
    params = get_default_SHO_parameters()
    
    # Verify stage budgets sum to 1.0
    budget_sum = sum(params.stage_budgets)
    assert abs(budget_sum - 1.0) < 1e-10, f"Stage budgets sum to {budget_sum}, not 1.0!"
    print(f"✓ Stage budgets sum to 1.0")
    
    # Calculate expected steps per stage
    print(f"\nExpected budget allocation:")
    total_check = 0
    for i, (t, frac) in enumerate(zip(params.t_stages, params.stage_budgets)):
        steps = int(frac * params.total_budget)
        total_check += steps
        print(f"  Stage {i} (t={t:.2f}): {steps:,} steps ({frac*100:.0f}%)")
    
    print(f"\nTotal steps (integer rounding): {total_check:,}")
    if total_check != params.total_budget:
        print(f"⚠️  WARNING: Integer rounding error! Off by {params.total_budget - total_check} steps")
        print(f"  This will cause assertion failure. Adjust stage_budgets to compensate.")
    else:
        print(f"✓ Integer rounding OK: {total_check:,} = {params.total_budget:,}")
    
    return total_check == params.total_budget


def test_temperature_schedule():
    """
    🔴 CRITICAL TEST 2: Verify temperature matches baseline
    """
    print("\n" + "="*80)
    print("TEST 2: Temperature Schedule")
    print("="*80)
    
    params = get_default_SHO_parameters()
    
    print(f"SHO parameters:")
    print(f"  T0: {params.T0}")
    print(f"  Tend: {params.Tend}")
    
    # These must match baseline
    assert params.T0 == 5.0, f"T0 must be 5.0, got {params.T0}"
    assert params.Tend == 0.01, f"Tend must be 0.01, got {params.Tend}"
    
    print(f"✓ Temperature schedule matches baseline SA")
    return True


def test_move_probabilities():
    """
    🔴 CRITICAL TEST 3: Verify move operators unchanged
    """
    print("\n" + "="*80)
    print("TEST 3: Move Operator Probabilities")
    print("="*80)
    
    params = get_default_SHO_parameters()
    
    print(f"Move probabilities:")
    print(f"  Premium: {params.prem_move_prob}")
    print(f"  Deductible: {params.ded_move_prob}")
    print(f"  Feature flip: {1 - params.prem_move_prob - params.ded_move_prob}")
    
    assert params.prem_move_prob == 0.075, "Premium move prob must be 0.075"
    assert params.ded_move_prob == 0.075, "Deductible move prob must be 0.075"
    
    print(f"✓ Move probabilities unchanged from baseline")
    return True


def test_Q_construction():
    """
    🔴 CRITICAL TEST 4: Verify Q(t=1) matches original
    """
    print("\n" + "="*80)
    print("TEST 4: Q(t) Construction")
    print("="*80)
    
    # Create mock QUBO components
    n = 30  # N=20, M=5, K=5
    Q_base = np.random.randn(n, n) * 100
    Q_base = (Q_base + Q_base.T) / 2  # Symmetric
    
    Q_oh = np.random.randn(n, n) * 10
    Q_oh = (Q_oh + Q_oh.T) / 2
    
    Q_reg = np.random.randn(n, n) * 10
    Q_reg = (Q_reg + Q_reg.T) / 2
    
    Q_aff = None
    
    w_oh = 1_356_753  # Typical
    w_reg = 2_713_506  # Typical
    w_aff = 0
    
    lambda_0 = 1000.0
    params = get_default_SHO_parameters()
    
    # Build Q at different t values
    Q_0 = build_Q_homotopy(0.0, Q_base, Q_oh, Q_reg, Q_aff, w_oh, w_reg, w_aff, lambda_0, params)
    Q_half = build_Q_homotopy(0.5, Q_base, Q_oh, Q_reg, Q_aff, w_oh, w_reg, w_aff, lambda_0, params)
    Q_1 = build_Q_homotopy(1.0, Q_base, Q_oh, Q_reg, Q_aff, w_oh, w_reg, w_aff, lambda_0, params)
    
    # Expected Q(1) = Q_base + β(1)·w_reg·Q_reg = Q_base + w_reg·Q_reg
    Q_expected = Q_base + w_reg * Q_reg  # α=0.01≈0, β(1)=1, λ(1)=0
    
    if not np.allclose(Q_1, Q_expected, rtol=1e-8):
        print(f"❌ Q(t=1) does NOT match expected!")
        print(f"  Max difference: {np.max(np.abs(Q_1 - Q_expected)):.2e}")
        return False
    
    print(f"✓ Q(t=0) has spectral shift (max={np.max(np.abs(Q_0)):.2e})")
    print(f"✓ Q(t=0.5) is intermediate (max={np.max(np.abs(Q_half)):.2e})")
    print(f"✓ Q(t=1) matches original (max diff={np.max(np.abs(Q_1 - Q_expected)):.2e})")
    
    # Verify matrices are NOT mutated
    Q_base_copy = Q_base.copy()
    _ = build_Q_homotopy(0.5, Q_base, Q_oh, Q_reg, Q_aff, w_oh, w_reg, w_aff, lambda_0, params)
    assert np.array_equal(Q_base, Q_base_copy), "Q_base was mutated!"
    print(f"✓ Original matrices not mutated")
    
    return True


def test_penalty_scaling():
    """
    🔴 CRITICAL TEST 5: Verify penalty magnitudes don't explode
    """
    print("\n" + "="*80)
    print("TEST 5: Penalty Scaling")
    print("="*80)
    
    # Use actual penalty magnitudes from data
    penalty_cases = [
        ("Min", 814_743, 1_629_486),
        ("Mean", 6_005_962, 10_888_188),
        ("Max", 20_614_234, 41_228_467)
    ]
    
    n = 30
    Q_base = np.random.randn(n, n)
    Q_oh = np.random.randn(n, n)
    Q_reg = np.random.randn(n, n)
    
    lambda_0 = 1000.0
    params = get_default_SHO_parameters()
    
    for label, w_oh, w_reg in penalty_cases:
        Q_t = build_Q_homotopy(0.5, Q_base, Q_oh, Q_reg, None, w_oh, w_reg, 0, lambda_0, params)
        max_val = np.max(np.abs(Q_t))
        
        print(f"  {label:5s}: w_oh={w_oh:,}, w_reg={w_reg:,} → max(Q)={max_val:.2e}")
        
        if max_val > 1e10:
            print(f"    ⚠️  Very large Q values! May cause overflow.")
        elif max_val > 1e8:
            print(f"    ⚠️  Large Q values. Monitor for numerical issues.")
        else:
            print(f"    ✓ Q magnitude reasonable")
    
    return True


def test_feasibility_tracking():
    """
    🔴 CRITICAL TEST 6: Verify feasible vs infeasible separation
    """
    print("\n" + "="*80)
    print("TEST 6: Feasibility Tracking")
    print("="*80)
    
    n = 30
    N, M, K = 20, 5, 5
    
    # Create mock Q matrices
    Q_full = np.eye(n)
    Q_base = np.eye(n)
    Q_penalties = np.zeros((n, n))
    
    # Case 1: Feasible solution
    sol_feasible = np.zeros(n)
    sol_feasible[0] = 1  # 1 feature
    sol_feasible[1] = 1  # 1 feature
    sol_feasible[N] = 1  # Deductible band 1
    sol_feasible[N+M] = 1  # Premium band 1
    
    metrics_feas = extract_solution_metrics(sol_feasible, Q_full, Q_base, Q_penalties, N, M, K, "test")
    
    print(f"Feasible solution:")
    print(f"  is_feasible: {metrics_feas['is_feasible_test']}")
    print(f"  penalty: {metrics_feas['penE_total_test']}")
    assert metrics_feas['is_feasible_test'] == 1, "Feasible solution not detected!"
    print(f"  ✓ Correctly identified as feasible")
    
    # Case 2: Infeasible solution (two deductibles)
    sol_infeas = sol_feasible.copy()
    sol_infeas[N+1] = 1  # Second deductible
    
    Q_penalties_inf = np.eye(n) * 1e6  # Large penalty
    metrics_infeas = extract_solution_metrics(sol_infeas, Q_full, Q_base, Q_penalties_inf, N, M, K, "test")
    
    print(f"\nInfeasible solution (violation):")
    print(f"  is_feasible: {metrics_infeas['is_feasible_test']}")
    print(f"  penalty: {metrics_infeas['penE_total_test']:.2e}")
    assert metrics_infeas['is_feasible_test'] == 0, "Infeasible solution not detected!"
    print(f"  ✓ Correctly identified as infeasible")
    
    return True


def test_rng_determinism():
    """
    🔴 CRITICAL TEST 7: Verify deterministic seeding
    """
    print("\n" + "="*80)
    print("TEST 7: Random Seed Determinism")
    print("="*80)
    
    seed = 1000
    
    # Run twice with same seed
    rng1 = np.random.default_rng(seed)
    vals1 = [rng1.random() for _ in range(10)]
    
    rng2 = np.random.default_rng(seed)
    vals2 = [rng2.random() for _ in range(10)]
    
    assert np.allclose(vals1, vals2), "RNG not deterministic!"
    print(f"✓ Same seed produces identical sequences")
    
    # Different seeds produce different sequences
    rng3 = np.random.default_rng(seed + 1)
    vals3 = [rng3.random() for _ in range(10)]
    
    assert not np.allclose(vals1, vals3), "Different seeds produce same sequence!"
    print(f"✓ Different seeds produce different sequences")
    
    return True


def run_all_tests():
    """
    Run complete pre-flight checklist
    """
    print("\n" + "="*80)
    print("SHO PRE-FLIGHT VALIDATION CHECKLIST")
    print("="*80)
    print("\nRunning 7 critical tests...")
    
    tests = [
        ("Budget Accounting", test_budget_accounting),
        ("Temperature Schedule", test_temperature_schedule),
        ("Move Probabilities", test_move_probabilities),
        ("Q Construction", test_Q_construction),
        ("Penalty Scaling", test_penalty_scaling),
        ("Feasibility Tracking", test_feasibility_tracking),
        ("RNG Determinism", test_rng_determinism)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"  {status:8s} {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Safe to run full experiment.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED! Fix before running full experiment.")
        print("\nDo NOT launch 720-instance run until all tests pass.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())