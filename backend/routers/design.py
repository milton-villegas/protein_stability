"""Design routes - factor CRUD, design generation, export"""

import logging
import traceback
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response

logger = logging.getLogger(__name__)

from backend.dependencies import get_current_session
from backend.schemas.factors import (
    FactorAddRequest, FactorUpdateRequest, FactorFromAvailableRequest,
)
from backend.schemas.design import (
    DesignGenerateRequest, BuildFactorialRequest, DesignGenerateResponse, ExportRequest,
)
from backend.services import design_service, export_service

router = APIRouter()


def _build_factors_response(project) -> dict:
    """Build standard factors response from project"""
    factors = project.get_factors()
    stock_concs = project.get_all_stock_concs()
    per_level = project.get_all_per_level_concs()
    total = design_service.get_combinations_count(factors)
    plates = design_service.get_plates_required(total)

    return {
        "factors": factors,
        "stock_concs": stock_concs,
        "per_level_concs": per_level,
        "total_combinations": total,
        "plates_required": plates,
    }


@router.get("/factors")
async def get_factors(session: dict = Depends(get_current_session)):
    """Get current project factors"""
    return _build_factors_response(session["project"])


@router.post("/factors")
async def add_factor(
    body: FactorAddRequest,
    session: dict = Depends(get_current_session),
):
    """Add a factor to the project"""
    project = session["project"]
    project.add_factor(body.name, body.levels, body.stock_conc)
    if body.per_level_concs:
        project.set_per_level_concs(body.name, body.per_level_concs)
    return _build_factors_response(project)


@router.post("/factors/from-available")
async def add_from_available(
    body: FactorFromAvailableRequest,
    session: dict = Depends(get_current_session),
):
    """Add a factor from the available factors list"""
    project = session["project"]
    project.add_factor(body.internal_name, body.levels, body.stock_conc)
    if body.per_level_concs:
        project.set_per_level_concs(body.internal_name, body.per_level_concs)
    return _build_factors_response(project)


@router.put("/factors/{name}")
async def update_factor(
    name: str,
    body: FactorUpdateRequest,
    session: dict = Depends(get_current_session),
):
    """Update an existing factor"""
    project = session["project"]
    project.update_factor(name, body.levels, body.stock_conc)
    if body.per_level_concs:
        project.set_per_level_concs(name, body.per_level_concs)
    return _build_factors_response(project)


@router.delete("/factors/{name}")
async def remove_factor(
    name: str,
    session: dict = Depends(get_current_session),
):
    """Remove a factor"""
    project = session["project"]
    project.remove_factor(name)
    return _build_factors_response(project)


@router.post("/factors/clear")
async def clear_factors(session: dict = Depends(get_current_session)):
    """Clear all factors"""
    session["project"].clear_factors()
    return _build_factors_response(session["project"])


@router.get("/combinations")
async def get_combinations(session: dict = Depends(get_current_session)):
    """Get combination count and plates required"""
    factors = session["project"].get_factors()
    total = design_service.get_combinations_count(factors)
    plates = design_service.get_plates_required(total)
    return {"total_combinations": total, "plates_required": plates}


@router.post("/generate")
async def generate_design(
    body: DesignGenerateRequest,
    session: dict = Depends(get_current_session),
):
    """Generate design using DesignFactory (any design type)"""
    project = session["project"]
    factory = session["design_factory"]
    factors = project.get_factors()

    if not factors:
        raise HTTPException(400, "No factors defined")

    stock_concs = project.get_all_stock_concs()

    # Store protein params in session
    if body.protein_stock is not None:
        session["protein_stock"] = body.protein_stock
    if body.protein_final is not None:
        session["protein_final"] = body.protein_final

    try:
        design_points, warnings = design_service.generate_design(
            factory, body.design_type, factors, body.params,
            stock_concs=stock_concs, final_volume=body.final_volume,
        )

        total_runs = len(design_points)
        plates = design_service.get_plates_required(total_runs)

        return DesignGenerateResponse(
            design_points=design_points,
            total_runs=total_runs,
            plates_required=plates,
            warnings=warnings,
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/build-factorial")
async def build_factorial(
    body: BuildFactorialRequest,
    session: dict = Depends(get_current_session),
):
    """Build full factorial design with volume calculations"""
    project = session["project"]
    designer = session["designer"]
    factors = project.get_factors()
    stock_concs = project.get_all_stock_concs()

    if not factors:
        raise HTTPException(400, "No factors defined")

    # Store protein params in session for volume calculations
    if body.protein_stock is not None:
        session["protein_stock"] = body.protein_stock
    if body.protein_final is not None:
        session["protein_final"] = body.protein_final

    try:
        logger.info(f"[BUILD-FACTORIAL] factors={list(factors.keys())}, stock_concs={stock_concs}, "
                     f"final_volume={body.final_volume}, protein_stock={body.protein_stock}, protein_final={body.protein_final}")
        per_level_concs = project.get_all_per_level_concs()
        excel_data, volume_data, warnings = design_service.build_factorial_design(
            designer, factors, stock_concs, body.final_volume,
            per_level_concs=per_level_concs,
            protein_stock=body.protein_stock,
            protein_final=body.protein_final,
        )
        logger.info(f"[BUILD-FACTORIAL] Success: {len(excel_data)} runs")
        return {
            "excel_data": excel_data,
            "volume_data": volume_data,
            "warnings": warnings,
            "total_runs": len(excel_data),
            "plates_required": design_service.get_plates_required(len(excel_data)),
        }
    except Exception as e:
        logger.error(f"[BUILD-FACTORIAL] ERROR: {e}\n{traceback.format_exc()}")
        raise HTTPException(400, str(e))


@router.post("/export/excel")
async def export_excel(
    body: ExportRequest,
    session: dict = Depends(get_current_session),
):
    """Export design as Excel file"""
    project = session["project"]
    designer = session["designer"]
    factors = project.get_factors()
    stock_concs = project.get_all_stock_concs()

    if not factors:
        raise HTTPException(400, "No factors defined")

    per_level_concs = project.get_all_per_level_concs()

    try:
        excel_df, volume_df = designer.build_factorial_design(
            factors, stock_concs, body.final_volume,
            per_level_concs=per_level_concs,
            protein_stock=body.protein_stock,
            protein_final=body.protein_final,
        )

        excel_bytes = export_service.generate_excel_bytes(
            excel_df, volume_df, stock_concs, project.name,
            per_level_concs=per_level_concs,
            protein_stock=body.protein_stock,
            protein_final=body.protein_final,
            final_volume=body.final_volume,
        )

        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"{project.name}_Design_{date_str}.xlsx"
        return Response(
            content=excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/export/csv")
async def export_csv(
    body: ExportRequest,
    session: dict = Depends(get_current_session),
):
    """Export design as Opentrons CSV"""
    project = session["project"]
    designer = session["designer"]
    factors = project.get_factors()
    stock_concs = project.get_all_stock_concs()

    if not factors:
        raise HTTPException(400, "No factors defined")

    per_level_concs = project.get_all_per_level_concs()

    try:
        _, volume_df = designer.build_factorial_design(
            factors, stock_concs, body.final_volume,
            per_level_concs=per_level_concs,
            protein_stock=body.protein_stock,
            protein_final=body.protein_final,
        )

        csv_bytes = export_service.generate_csv_bytes(volume_df)

        date_str = datetime.now().strftime('%Y%m%d')
        filename = f"{project.name}_Design_{date_str}_Opentron.csv"
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/validate")
async def validate_design(
    body: dict,
    session: dict = Depends(get_current_session),
):
    """Validate design parameters"""
    project = session["project"]
    factors = project.get_factors()
    design_type = body.get("design_type", "full_factorial")

    valid, errors, warnings = design_service.validate_design_params(
        design_type, factors,
        has_pydoe3=session.get("has_pydoe3", False),
        has_smt=session.get("has_smt", False),
    )

    return {"valid": valid, "errors": errors, "warnings": warnings}
