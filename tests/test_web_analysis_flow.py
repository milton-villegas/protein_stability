"""
Test the full web analysis flow end-to-end without running the server.
Simulates: upload → configure → run → plots → compare_models → optimize → serialize
Verifies JSON keys match frontend expectations.

Run with: PYTHONPATH=. python tests/test_web_analysis_flow.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def test_make_serializable():
    """Test _make_serializable handles all edge cases"""
    from backend.services.analysis_service import _make_serializable

    print("=== Testing _make_serializable ===")

    assert _make_serializable(42) == 42
    assert _make_serializable("hello") == "hello"
    assert _make_serializable(np.int64(5)) == 5
    assert _make_serializable(np.float64(2.5)) == 2.5
    assert _make_serializable(np.bool_(True)) is True
    assert _make_serializable(np.float64("nan")) is None
    print("  [PASS] Scalar types")

    zero_d = np.array(3.14)
    assert _make_serializable(zero_d) == 3.14
    assert _make_serializable(np.array([1.0, 2.0])) == [1.0, 2.0]
    print("  [PASS] numpy arrays (0-d and 1-d)")

    df = pd.DataFrame({"a": [1, 2], "b": [3.0, np.nan]})
    result = _make_serializable(df)
    assert isinstance(result, list) and len(result) == 2
    json.dumps(result)
    print("  [PASS] DataFrame")

    s = pd.Series([1.0, 2.0, np.nan], index=["x", "y", "z"])
    result = _make_serializable(s)
    assert isinstance(result, dict) and result["z"] is None
    json.dumps(result)
    print("  [PASS] Series")

    class FakeModel:
        pass
    assert _make_serializable(FakeModel()) is None
    assert _make_serializable(float("nan")) is None
    assert _make_serializable(float("inf")) is None
    print("  [PASS] Non-serializable / NaN / Inf")

    nested = {"stats": {"r2": np.float64(0.95)}, "values": np.array([1.0])}
    parsed = json.loads(json.dumps(_make_serializable(nested)))
    assert parsed["stats"]["r2"] == 0.95
    print("  [PASS] Nested dict round-trip")
    print()


def test_key_normalization():
    """Test key normalization matches frontend expectations"""
    from backend.services.analysis_service import _normalize_key

    print("=== Testing Key Normalization ===")

    # _extract_results keys
    assert _normalize_key("R-squared") == "r_squared"
    assert _normalize_key("Adjusted R-squared") == "adj_r_squared"
    assert _normalize_key("F-statistic") == "f_statistic"
    assert _normalize_key("F p-value") == "f_pvalue"
    assert _normalize_key("RMSE") == "rmse"
    assert _normalize_key("AIC") == "aic"
    assert _normalize_key("BIC") == "bic"
    assert _normalize_key("Observations") == "observations"
    assert _normalize_key("DF Residuals") == "df_residuals"
    assert _normalize_key("DF Model") == "df_model"
    print("  [PASS] model_stats keys")

    # compare_all_models keys
    assert _normalize_key("Model Type") == "model_type"
    r2_key = _normalize_key("R²")
    assert "r" in r2_key and "squared" in r2_key
    adj_key = _normalize_key("Adj R²")
    assert "adj" in adj_key
    print("  [PASS] compare_models keys")
    print()


def test_full_analysis_flow():
    """Test: load → configure → run → compare → effects → plots → optimize"""
    from backend.services import analysis_service, plot_service, optimization_service

    print("=== Testing Full Analysis Flow ===")

    test_file = "examples/test_multi_response_data.xlsx"
    if not os.path.exists(test_file):
        print(f"  [SKIP] {test_file} not found")
        return

    # --- 1. Load ---
    print("  1. load_and_detect")
    result = analysis_service.load_and_detect(test_file)
    handler = result.pop("handler")
    json.dumps(result)
    print(f"     {result['total_rows']} rows, {len(result['columns'])} cols")
    print(f"     Responses: {result['potential_responses']}")

    # --- 2. Configure ---
    print("  2. configure_analysis")
    responses = result["potential_responses"][:2]  # Use 2 responses for multi-response testing
    config = analysis_service.configure_analysis(handler, responses)
    json.dumps(config)
    print(f"     Factors: {config['factor_columns']}")

    # --- 3. Analyzer setup ---
    print("  3. DoEAnalyzer setup")
    from core.doe_analyzer import DoEAnalyzer
    analyzer = DoEAnalyzer()
    analyzer.set_data(
        data=handler.clean_data,
        factor_columns=handler.factor_columns,
        categorical_factors=handler.categorical_factors,
        numeric_factors=handler.numeric_factors,
        response_columns=responses,
    )

    # --- 4. Run analysis ---
    print("  4. run_analysis")
    results = analysis_service.run_analysis(analyzer, "linear")
    json_str = json.dumps(results)

    for resp_name, resp_data in results.items():
        # model_stats keys
        stats = resp_data.get("model_stats", {})
        for key in ["r_squared", "adj_r_squared", "f_statistic", "f_pvalue"]:
            assert key in stats, f"Missing '{key}' in model_stats. Keys: {list(stats.keys())}"
            assert isinstance(stats[key], (int, float)), f"'{key}' is {type(stats[key]).__name__}"
        # coefficients
        coeffs = resp_data.get("coefficients")
        assert isinstance(coeffs, list) and len(coeffs) > 0 and isinstance(coeffs[0], dict)
        # model_object stripped
        assert resp_data.get("model_object") is None
        print(f"     {resp_name}: r²={stats['r_squared']:.4f}, adj_r²={stats['adj_r_squared']:.4f}, F={stats['f_statistic']:.2f}")

    print(f"     JSON: {len(json_str)} bytes")

    # --- 5. Compare models ---
    print("  5. compare_models")
    comparison = analysis_service.compare_models(analyzer)
    json_str = json.dumps(comparison)

    assert "comparisons" in comparison
    assert "recommendations" in comparison

    for resp_name, models in comparison["comparisons"].items():
        assert isinstance(models, dict), f"Expected dict, got {type(models).__name__}"
        for model_name, model_stats in models.items():
            assert isinstance(model_stats, dict)
            for v in model_stats.values():
                assert isinstance(v, (int, float, str, type(None))), \
                    f"{model_name} has non-scalar: {type(v).__name__}"

        rec = comparison["recommendations"][resp_name]
        assert isinstance(rec, dict), f"Recommendation should be dict, got {type(rec).__name__}"
        assert "best_model" in rec, f"Missing 'best_model'. Keys: {list(rec.keys())}"
        assert "reason" in rec, f"Missing 'reason'. Keys: {list(rec.keys())}"
        assert "scores" in rec, f"Missing 'scores'. Keys: {list(rec.keys())}"
        assert isinstance(rec["reason"], str) and len(rec["reason"]) > 0, "Reason should be non-empty string"
        assert isinstance(rec["scores"], dict) and len(rec["scores"]) > 0, "Scores should be non-empty dict"

        # Verify scores have expected keys
        for score_model, score_data in rec["scores"].items():
            for key in ["combined_score", "adj_r2", "bic", "rmse"]:
                assert key in score_data, f"Score for '{score_model}' missing '{key}'"

        print(f"     {resp_name}: best={rec['best_model']}, reason={rec['reason']}")
        print(f"       Scores: {list(rec['scores'].keys())}")

    print(f"     JSON: {len(json_str)} bytes")

    # --- 6. Main effects ---
    print("  6. get_main_effects")
    effects = analysis_service.get_main_effects(analyzer)
    json.dumps(effects)
    print(f"     Factors: {list(effects.keys())}")

    # --- 7. Plots ---
    print("  7. Plots")
    from core.plotter import DoEPlotter
    plotter = DoEPlotter()
    plotter.set_data(handler.clean_data, handler.factor_columns, responses[0])

    all_res = getattr(analyzer, "all_results", None) or {}
    resp_results = all_res.get(responses[0]) or getattr(analyzer, "results", None) or {}
    predictions = resp_results.get("predictions")
    residuals = resp_results.get("residuals")

    for name, gen_fn in [
        ("main-effects", lambda: plot_service.generate_main_effects_plot(plotter)),
        ("interactions", lambda: plot_service.generate_interaction_plot(plotter)),
        ("residuals", lambda: plot_service.generate_residuals_plot(plotter, predictions, residuals)),
        ("predictions", lambda: plot_service.generate_predictions_plot(plotter, predictions, residuals)),
    ]:
        img = gen_fn()
        assert img.startswith("data:image/png;base64,")
        print(f"     {name}: OK ({len(img)} chars)")

    # --- 8. Bayesian Optimization ---
    print("  8. Bayesian Optimization")
    session = {
        "data_handler": handler,
        "analyzer": analyzer,
    }
    directions = {r: "maximize" for r in responses}
    try:
        opt_result = optimization_service.initialize_optimizer(
            session, responses, directions, n_suggestions=3,
        )
        json_str = json.dumps(opt_result)
        assert "suggestions" in opt_result, f"Missing 'suggestions'. Keys: {list(opt_result.keys())}"
        assert isinstance(opt_result["suggestions"], list)
        assert len(opt_result["suggestions"]) > 0, "No suggestions returned"
        print(f"     {len(opt_result['suggestions'])} suggestions, pareto={opt_result.get('has_pareto')}")
        print(f"     First suggestion keys: {list(opt_result['suggestions'][0].keys())}")
        print(f"     JSON: {len(json_str)} bytes")
    except Exception as e:
        print(f"     [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise

    # --- 9. Analysis Summary ---
    print("  9. Analysis Summary")
    directions = {r: "maximize" for r in responses}
    summary = analysis_service.get_analysis_summary(analyzer, handler, directions)
    json_str = json.dumps(summary)

    for resp_name in responses:
        s = summary[resp_name]
        for key in ["r_squared", "n_observations", "n_significant", "warnings",
                     "confidence", "significant_factors", "interactions",
                     "optimal_directions", "best_experiment", "next_steps"]:
            assert key in s, f"Summary missing '{key}'. Keys: {list(s.keys())}"
        assert s["confidence"] in ("HIGH", "MEDIUM", "LOW"), f"Bad confidence: {s['confidence']}"
        assert isinstance(s["warnings"], list)
        assert isinstance(s["significant_factors"], list)
        assert isinstance(s["next_steps"], list) and len(s["next_steps"]) > 0
        if s["best_experiment"]:
            assert "value" in s["best_experiment"]
            assert "conditions" in s["best_experiment"]
        print(f"     {resp_name}: confidence={s['confidence']}, "
              f"n_sig={s['n_significant']}, warnings={len(s['warnings'])}")

    print(f"     JSON: {len(json_str)} bytes")

    print("\n  All tests passed!\n")


def test_multi_response_export():
    """Test that multi-response export doesn't crash (checks all_results path)"""
    from backend.services import analysis_service
    from core.doe_analyzer import DoEAnalyzer

    print("=== Testing Multi-Response Export ===")

    test_file = "examples/test_multi_response_data.xlsx"
    if not os.path.exists(test_file):
        print(f"  [SKIP] {test_file} not found")
        return

    result = analysis_service.load_and_detect(test_file)
    handler = result.pop("handler")
    responses = result["potential_responses"][:2]
    analysis_service.configure_analysis(handler, responses)

    analyzer = DoEAnalyzer()
    analyzer.set_data(
        data=handler.clean_data,
        factor_columns=handler.factor_columns,
        categorical_factors=handler.categorical_factors,
        numeric_factors=handler.numeric_factors,
        response_columns=responses,
    )
    analysis_service.run_analysis(analyzer, "linear")

    # Check that all_results is populated for multi-response
    all_results = getattr(analyzer, "all_results", None) or {}
    if not all_results:
        # Fall back to single-response
        assert analyzer.results is not None, "No results at all"
        all_results = {responses[0]: analyzer.results}

    assert len(all_results) >= 1, f"Expected at least 1 result, got {len(all_results)}"
    for resp_name, res in all_results.items():
        assert isinstance(res, dict), f"Result for {resp_name} should be dict"
        print(f"  {resp_name}: keys={list(res.keys())[:5]}")

    # Test main effects per response
    for resp_name in all_results:
        effects = analyzer.calculate_main_effects(response_name=resp_name)
        assert isinstance(effects, dict)
        print(f"  {resp_name} effects: {list(effects.keys())}")

    print("  [PASS] Multi-response export")
    print()


def test_bo_export():
    """Test that BO export uses last_suggestions correctly"""
    from backend.services import analysis_service, optimization_service
    from core.doe_analyzer import DoEAnalyzer

    print("=== Testing BO Export ===")

    test_file = "examples/test_multi_response_data.xlsx"
    if not os.path.exists(test_file):
        print(f"  [SKIP] {test_file} not found")
        return

    result = analysis_service.load_and_detect(test_file)
    handler = result.pop("handler")
    responses = result["potential_responses"][:1]
    analysis_service.configure_analysis(handler, responses)

    analyzer = DoEAnalyzer()
    analyzer.set_data(
        data=handler.clean_data,
        factor_columns=handler.factor_columns,
        categorical_factors=handler.categorical_factors,
        numeric_factors=handler.numeric_factors,
        response_columns=responses,
    )
    analysis_service.run_analysis(analyzer, "linear")

    session = {"data_handler": handler, "analyzer": analyzer}
    directions = {r: "maximize" for r in responses}

    try:
        opt_result = optimization_service.initialize_optimizer(
            session, responses, directions, n_suggestions=3,
        )
        optimizer = session["optimizer"]

        # Verify last_suggestions is stored
        assert hasattr(optimizer, "last_suggestions"), "last_suggestions not set on optimizer"
        assert len(optimizer.last_suggestions) == 3
        print(f"  last_suggestions: {len(optimizer.last_suggestions)} items")

        # Verify we can build a DataFrame from them (the export path)
        import pandas as pd
        df = pd.DataFrame(optimizer.last_suggestions)
        assert len(df) == 3
        print(f"  DataFrame columns: {list(df.columns)}")
        print("  [PASS] BO export")
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise

    print()


def test_pareto_endpoint():
    """Test Pareto frontier with multi-objective optimization"""
    from backend.services import analysis_service, optimization_service
    from core.doe_analyzer import DoEAnalyzer

    print("=== Testing Pareto Endpoint ===")

    test_file = "examples/test_multi_response_data.xlsx"
    if not os.path.exists(test_file):
        print(f"  [SKIP] {test_file} not found")
        return

    result = analysis_service.load_and_detect(test_file)
    handler = result.pop("handler")
    responses = result["potential_responses"][:2]

    if len(responses) < 2:
        print("  [SKIP] Need 2+ responses for multi-objective")
        return

    analysis_service.configure_analysis(handler, responses)

    analyzer = DoEAnalyzer()
    analyzer.set_data(
        data=handler.clean_data,
        factor_columns=handler.factor_columns,
        categorical_factors=handler.categorical_factors,
        numeric_factors=handler.numeric_factors,
        response_columns=responses,
    )
    analysis_service.run_analysis(analyzer, "linear")

    session = {"data_handler": handler, "analyzer": analyzer}
    directions = {r: "maximize" for r in responses}

    try:
        opt_result = optimization_service.initialize_optimizer(
            session, responses, directions, n_suggestions=3,
        )
        optimizer = session["optimizer"]

        assert optimizer.is_multi_objective, "Should be multi-objective"

        pareto_points = optimizer.get_pareto_frontier()
        assert isinstance(pareto_points, list), f"Expected list, got {type(pareto_points)}"
        print(f"  Pareto points: {len(pareto_points)}")
        if pareto_points:
            first = pareto_points[0]
            assert "parameters" in first, f"Missing 'parameters'. Keys: {list(first.keys())}"
            assert "objectives" in first, f"Missing 'objectives'. Keys: {list(first.keys())}"
            print(f"  First point keys: {list(first.keys())}")

            # Verify JSON serializable
            from backend.services.analysis_service import _make_serializable
            json.dumps(_make_serializable(pareto_points))
            print("  JSON serializable: OK")

        print("  [PASS] Pareto endpoint")
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        raise

    print()


if __name__ == "__main__":
    print()
    passed = True

    for test_fn in [
        test_make_serializable,
        test_key_normalization,
        test_full_analysis_flow,
        test_multi_response_export,
        test_bo_export,
        test_pareto_endpoint,
    ]:
        try:
            test_fn()
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            passed = False

    print("=" * 50)
    if passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 50)
