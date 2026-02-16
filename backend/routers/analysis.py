"""Analysis routes - upload, configure, run analysis, plots, optimization"""

import json
import logging
import os
import tempfile
import traceback

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import JSONResponse, Response
from typing import Optional

logger = logging.getLogger(__name__)

from backend.dependencies import get_current_session
from backend.schemas.analysis import (
    AnalysisConfigureRequest, AnalysisRunRequest,
    OptimizeRequest, UploadResponse, BOExportRequest,
)
from backend.services import analysis_service, plot_service, optimization_service, export_service
from backend.services.analysis_service import _make_serializable

router = APIRouter()


def safe_json_response(data: dict) -> JSONResponse:
    """Return a JSONResponse after converting all numpy/pandas types"""
    clean = _make_serializable(data)
    return JSONResponse(content=clean)


@router.post("/upload")
async def upload_results(
    file: UploadFile = File(...),
    session: dict = Depends(get_current_session),
):
    """Upload Excel results file"""
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(400, "Only Excel files (.xlsx, .xls) are supported")

    # Write to temp file (DataHandler expects file path)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        logger.info(f"[UPLOAD] Loading file: {tmp_path}")
        result = analysis_service.load_and_detect(tmp_path)
        handler = result.pop("handler")
        session["data_handler"] = handler
        session["upload_path"] = tmp_path
        logger.info(f"[UPLOAD] Success: {result['total_rows']} rows, {len(result['columns'])} columns")

        return UploadResponse(
            columns=result["columns"],
            potential_responses=result["potential_responses"],
            factor_columns=[],
            preview_rows=result["preview_rows"],
            total_rows=result["total_rows"],
        )
    except Exception as e:
        logger.error(f"[UPLOAD] ERROR: {e}\n{traceback.format_exc()}")
        os.unlink(tmp_path)
        raise HTTPException(400, f"Failed to load file: {str(e)}")


@router.post("/configure")
async def configure_analysis(
    body: AnalysisConfigureRequest,
    session: dict = Depends(get_current_session),
):
    """Configure response columns and analysis settings"""
    handler = session.get("data_handler")
    if handler is None:
        raise HTTPException(400, "No data uploaded. Upload an Excel file first.")

    try:
        logger.info(f"[CONFIGURE] response_columns={body.response_columns}, directions={body.directions}")
        result = analysis_service.configure_analysis(handler, body.response_columns)

        # Set up analyzer with configured data
        analyzer = session["analyzer"]
        logger.info(f"[CONFIGURE] Setting analyzer data: factors={handler.factor_columns}, "
                     f"categorical={handler.categorical_factors}, numeric={handler.numeric_factors}")
        analyzer.set_data(
            data=handler.clean_data,
            factor_columns=handler.factor_columns,
            categorical_factors=handler.categorical_factors,
            numeric_factors=handler.numeric_factors,
            response_columns=body.response_columns,
        )

        # Store directions and constraints for optimization
        if body.directions:
            session["response_directions"] = body.directions
        if body.constraints:
            session["response_constraints"] = body.constraints

        logger.info(f"[CONFIGURE] Success: {result}")
        return safe_json_response(result)
    except Exception as e:
        logger.error(f"[CONFIGURE] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.post("/run")
async def run_analysis(
    body: AnalysisRunRequest,
    session: dict = Depends(get_current_session),
):
    """Run statistical analysis"""
    analyzer = session["analyzer"]
    if analyzer.data is None:
        raise HTTPException(400, "No data configured. Configure analysis first.")

    try:
        logger.info(f"[RUN] model_type={body.model_type}, response_columns={analyzer.response_columns}")
        logger.info(f"[RUN] analyzer.data shape={analyzer.data.shape}, factors={analyzer.factor_columns}")
        results = analysis_service.run_analysis(analyzer, body.model_type)
        logger.info(f"[RUN] Analysis complete, result keys per response: "
                     f"{ {k: list(v.keys()) if isinstance(v, dict) else type(v).__name__ for k, v in results.items()} }")

        # Set up plotter with the data
        handler = session.get("data_handler")
        if handler and handler.clean_data is not None:
            plotter = session["plotter"]
            response_col = (
                analyzer.response_columns[0]
                if analyzer.response_columns
                else analyzer.response_column
            )
            logger.info(f"[RUN] Setting plotter data, response_col={response_col}")
            plotter.set_data(
                handler.clean_data,
                handler.factor_columns,
                response_col,
            )

        logger.info("[RUN] Building safe JSON response...")
        response = safe_json_response({"results": results})
        logger.info("[RUN] Success")
        return response
    except Exception as e:
        logger.error(f"[RUN] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.post("/compare-models")
async def compare_models(session: dict = Depends(get_current_session)):
    """Compare all model types"""
    analyzer = session["analyzer"]
    if analyzer.data is None:
        raise HTTPException(400, "No data configured")

    try:
        logger.info("[COMPARE] Starting model comparison")
        result = analysis_service.compare_models(analyzer)
        logger.info(f"[COMPARE] Success, recommendations={result.get('recommendations', {})}")
        return safe_json_response(result)
    except Exception as e:
        logger.error(f"[COMPARE] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.get("/summary")
async def get_analysis_summary(
    session: dict = Depends(get_current_session),
):
    """Get analysis summary with warnings, significant factors, recommendations"""
    analyzer = session["analyzer"]
    if analyzer.data is None:
        raise HTTPException(400, "No data configured")

    try:
        logger.info("[SUMMARY] Generating analysis summary")
        handler = session.get("data_handler")
        directions = session.get("response_directions", {})
        result = analysis_service.get_analysis_summary(analyzer, handler, directions)
        logger.info(f"[SUMMARY] Success, responses={list(result.keys())}")
        return safe_json_response(result)
    except Exception as e:
        logger.error(f"[SUMMARY] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.get("/main-effects")
async def get_main_effects(
    response: Optional[str] = Query(None),
    session: dict = Depends(get_current_session),
):
    """Get main effects data"""
    analyzer = session["analyzer"]
    if analyzer.data is None:
        raise HTTPException(400, "No data configured")

    try:
        logger.info(f"[MAIN-EFFECTS] response={response}")
        effects = analysis_service.get_main_effects(analyzer, response)
        logger.info(f"[MAIN-EFFECTS] Success, factors={list(effects.keys())}")
        return safe_json_response({"effects": effects})
    except Exception as e:
        logger.error(f"[MAIN-EFFECTS] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.get("/plot/{plot_type}")
async def get_plot(
    plot_type: str,
    response: Optional[str] = Query(None),
    session: dict = Depends(get_current_session),
):
    """Get analysis plot as base64 image"""
    plotter = session["plotter"]

    valid_types = ["main-effects", "interactions", "residuals", "predictions", "distribution", "qq",
                    "bo-response-surface", "bo-pareto", "bo-parallel"]
    if plot_type not in valid_types:
        raise HTTPException(400, f"Unknown plot type: {plot_type}. Available: {valid_types}")

    # BO-specific plots delegate to optimizer
    if plot_type.startswith("bo-"):
        optimizer = session.get("optimizer")
        if optimizer is None:
            raise HTTPException(400, "No optimization results. Run Bayesian Optimization first.")
        try:
            if plot_type == "bo-response-surface":
                fig = optimizer.get_acquisition_plot()
            elif plot_type == "bo-pareto":
                fig = optimizer.plot_pareto_frontier()
            elif plot_type == "bo-parallel":
                fig = optimizer.plot_pareto_parallel_coordinates()
            else:
                fig = None
            if fig is None:
                raise HTTPException(400, f"Plot '{plot_type}' not available for this optimization configuration")
            image = plot_service.figure_to_base64(fig)
            return {"image": image, "plot_type": plot_type}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[PLOT] ERROR generating {plot_type}: {e}\n{traceback.format_exc()}")
            raise HTTPException(400, f"Failed to generate plot: {str(e)}")

    try:
        logger.info(f"[PLOT] Generating plot_type={plot_type}, response={response}")

        # Update plotter data if a specific response is requested
        analyzer = session.get("analyzer")
        handler = session.get("data_handler")
        if response and handler and handler.clean_data is not None:
            plotter.set_data(handler.clean_data, handler.factor_columns, response)

        # Get predictions/residuals from analyzer for plots that need them
        predictions = None
        residuals = None
        if analyzer:
            resp_key = response or (analyzer.response_columns[0] if analyzer.response_columns else analyzer.response_column)
            # all_results stores per-response results from fit_model_all_responses
            all_res = getattr(analyzer, "all_results", None) or {}
            # single response stored in analyzer.results
            single_res = getattr(analyzer, "results", None) or {}
            resp_results = all_res.get(resp_key) or single_res
            predictions = resp_results.get("predictions") if resp_results else None
            residuals = resp_results.get("residuals") if resp_results else None
            logger.info(f"[PLOT] predictions available={predictions is not None}, residuals available={residuals is not None}")

        if plot_type == "main-effects":
            image = plot_service.generate_main_effects_plot(plotter)
        elif plot_type == "interactions":
            image = plot_service.generate_interaction_plot(plotter)
        elif plot_type == "residuals":
            image = plot_service.generate_residuals_plot(plotter, predictions, residuals)
        elif plot_type == "predictions":
            image = plot_service.generate_predictions_plot(plotter, predictions, residuals)
        elif plot_type == "distribution":
            image = plot_service.generate_response_distribution_plot(plotter)
        elif plot_type == "qq":
            image = plot_service.generate_qq_plot(plotter)

        logger.info(f"[PLOT] Success, image length={len(image)}")
        return {"image": image, "plot_type": plot_type}
    except Exception as e:
        logger.error(f"[PLOT] ERROR generating {plot_type}: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, f"Failed to generate plot: {str(e)}")


@router.post("/optimize")
async def run_optimization(
    body: OptimizeRequest,
    session: dict = Depends(get_current_session),
):
    """Run Bayesian optimization"""
    try:
        logger.info(f"[OPTIMIZE] responses={body.response_columns}, directions={body.directions}")
        result = optimization_service.initialize_optimizer(
            session,
            body.response_columns,
            body.directions,
            body.constraints,
            body.n_suggestions,
        )
        logger.info(f"[OPTIMIZE] Success, {len(result.get('suggestions', []))} suggestions")
        return safe_json_response(result)
    except Exception as e:
        logger.error(f"[OPTIMIZE] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.get("/export/results")
async def export_results(session: dict = Depends(get_current_session)):
    """Export analysis results as Excel file"""
    analyzer = session["analyzer"]

    # Support both single-response and multi-response
    all_results = getattr(analyzer, "all_results", None) or {}
    if not all_results:
        if analyzer.results is None:
            raise HTTPException(400, "No analysis results. Run analysis first.")
        resp_name = (
            analyzer.response_columns[0]
            if analyzer.response_columns
            else analyzer.response_column
        )
        all_results = {resp_name: analyzer.results}

    exporter = session["exporter"]

    try:
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name

        # Export each response's results
        for resp_name, results in all_results.items():
            effects = analyzer.calculate_main_effects(response_name=resp_name)
            exporter.set_results(results, effects)
            exporter.export_statistics_excel(tmp_path)

        with open(tmp_path, "rb") as f:
            content = f.read()
        os.unlink(tmp_path)

        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="analysis_results.xlsx"'},
        )
    except Exception as e:
        logger.error(f"[EXPORT-RESULTS] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.post("/export/bo-batch")
async def export_bo_batch(
    body: BOExportRequest,
    session: dict = Depends(get_current_session),
):
    """Export BO suggestions as Opentron-compatible Excel (3 sheets)"""
    optimizer = session.get("optimizer")
    if optimizer is None:
        raise HTTPException(400, "No optimization results. Run Bayesian Optimization first.")

    suggestions = getattr(optimizer, "last_suggestions", None)
    if not suggestions:
        raise HTTPException(400, "No suggestions available. Run optimization first.")

    handler = session.get("data_handler")
    if handler is None:
        raise HTTPException(400, "No data uploaded.")

    try:
        stock_concs = handler.get_stock_concentrations()
        per_level_concs = handler.get_per_level_concs()
        existing_data = handler.data

        excel_df, volume_df = export_service.build_bo_volume_data(
            suggestions, stock_concs, per_level_concs,
            body.final_volume, body.batch_number, existing_data,
        )

        content = export_service.generate_excel_bytes(
            excel_df, volume_df, stock_concs,
            project_name="BO_Batch",
            per_level_concs=per_level_concs,
            protein_stock=body.protein_stock,
            protein_final=body.protein_final,
            final_volume=body.final_volume,
        )

        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": 'attachment; filename="bo_batch.xlsx"'},
        )
    except Exception as e:
        logger.error(f"[EXPORT-BO] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.post("/export/bo-csv")
async def export_bo_csv(
    body: BOExportRequest,
    session: dict = Depends(get_current_session),
):
    """Export BO suggestions as Opentron-compatible CSV (volumes only)"""
    optimizer = session.get("optimizer")
    if optimizer is None:
        raise HTTPException(400, "No optimization results. Run Bayesian Optimization first.")

    suggestions = getattr(optimizer, "last_suggestions", None)
    if not suggestions:
        raise HTTPException(400, "No suggestions available. Run optimization first.")

    handler = session.get("data_handler")
    if handler is None:
        raise HTTPException(400, "No data uploaded.")

    try:
        stock_concs = handler.get_stock_concentrations()
        per_level_concs = handler.get_per_level_concs()
        existing_data = handler.data

        _, volume_df = export_service.build_bo_volume_data(
            suggestions, stock_concs, per_level_concs,
            body.final_volume, body.batch_number, existing_data,
        )

        content = export_service.generate_csv_bytes(volume_df)

        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="bo_batch.csv"'},
        )
    except Exception as e:
        logger.error(f"[EXPORT-BO-CSV] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.get("/pareto")
async def get_pareto_frontier(session: dict = Depends(get_current_session)):
    """Get Pareto frontier for multi-objective optimization"""
    optimizer = session.get("optimizer")
    if optimizer is None:
        raise HTTPException(400, "No optimization results. Run Bayesian Optimization first.")

    if not getattr(optimizer, "is_multi_objective", False):
        raise HTTPException(400, "Pareto frontier only available for multi-objective optimization")

    try:
        pareto_points = optimizer.get_pareto_frontier()
        return safe_json_response({"pareto_points": pareto_points})
    except Exception as e:
        logger.error(f"[PARETO] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))
