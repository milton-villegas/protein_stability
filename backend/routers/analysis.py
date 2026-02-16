"""Analysis routes - upload, configure, run analysis, plots, optimization"""

import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import Response
from typing import Optional

from backend.dependencies import get_current_session
from backend.schemas.analysis import (
    AnalysisConfigureRequest, AnalysisRunRequest, AnalysisRunResponse,
    OptimizeRequest, OptimizeResponse, UploadResponse,
)
from backend.services import analysis_service, plot_service, optimization_service

router = APIRouter()


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
        result = analysis_service.load_and_detect(tmp_path)
        handler = result.pop("handler")
        session["data_handler"] = handler
        session["upload_path"] = tmp_path

        return UploadResponse(
            columns=result["columns"],
            potential_responses=result["potential_responses"],
            factor_columns=[],
            preview_rows=result["preview_rows"],
            total_rows=result["total_rows"],
        )
    except Exception as e:
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
        result = analysis_service.configure_analysis(handler, body.response_columns)

        # Set up analyzer with configured data
        analyzer = session["analyzer"]
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

        return result
    except Exception as e:
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
        results = analysis_service.run_analysis(analyzer, body.model_type)

        # Set up plotter with the data
        handler = session.get("data_handler")
        if handler and handler.clean_data is not None:
            plotter = session["plotter"]
            response_col = (
                analyzer.response_columns[0]
                if analyzer.response_columns
                else analyzer.response_column
            )
            plotter.set_data(
                handler.clean_data,
                handler.factor_columns,
                response_col,
            )

        return {"results": results}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/compare-models")
async def compare_models(session: dict = Depends(get_current_session)):
    """Compare all model types"""
    analyzer = session["analyzer"]
    if analyzer.data is None:
        raise HTTPException(400, "No data configured")

    try:
        return analysis_service.compare_models(analyzer)
    except Exception as e:
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
        effects = analysis_service.get_main_effects(analyzer, response)
        return {"effects": effects}
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/plot/{plot_type}")
async def get_plot(
    plot_type: str,
    response: Optional[str] = Query(None),
    session: dict = Depends(get_current_session),
):
    """Get analysis plot as base64 image"""
    plotter = session["plotter"]

    plot_generators = {
        "main-effects": plot_service.generate_main_effects_plot,
        "interactions": plot_service.generate_interaction_plot,
        "residuals": plot_service.generate_residuals_plot,
        "predictions": plot_service.generate_predictions_plot,
        "distribution": plot_service.generate_response_distribution_plot,
        "qq": plot_service.generate_qq_plot,
    }

    if plot_type not in plot_generators:
        raise HTTPException(400, f"Unknown plot type: {plot_type}. Available: {list(plot_generators.keys())}")

    try:
        image = plot_generators[plot_type](plotter)
        return {"image": image, "plot_type": plot_type}
    except Exception as e:
        raise HTTPException(400, f"Failed to generate plot: {str(e)}")


@router.post("/optimize")
async def run_optimization(
    body: OptimizeRequest,
    session: dict = Depends(get_current_session),
):
    """Run Bayesian optimization"""
    try:
        result = optimization_service.initialize_optimizer(
            session,
            body.response_columns,
            body.directions,
            body.constraints,
            body.n_suggestions,
        )
        return OptimizeResponse(**result)
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/export/results")
async def export_results(session: dict = Depends(get_current_session)):
    """Export analysis results as Excel file"""
    analyzer = session["analyzer"]
    if analyzer.results is None:
        raise HTTPException(400, "No analysis results. Run analysis first.")

    exporter = session["exporter"]

    try:
        effects = analyzer.calculate_main_effects()
        exporter.set_results(analyzer.results, effects)

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp_path = tmp.name
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
        raise HTTPException(400, str(e))
