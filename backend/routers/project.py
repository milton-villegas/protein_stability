"""Project management routes"""

import io
import json
import tempfile
import os

from fastapi import APIRouter, Depends, Request, Response, UploadFile, File
from fastapi.responses import StreamingResponse

from backend.sessions import create_session
from backend.dependencies import get_current_session
from backend.schemas.project import ProjectCreateRequest, ProjectInfoResponse
from core.project import DoEProject

router = APIRouter()


@router.post("/new")
async def new_project(request_body: ProjectCreateRequest, request: Request, response: Response):
    """Create a new project and session"""
    session_id = create_session(request_body.name)
    # Store on request.state so middleware sets the header
    request.state.new_session_id = session_id
    response.headers["X-Session-ID"] = session_id
    return {"session_id": session_id, "name": request_body.name}


@router.get("/info")
async def get_project_info(session: dict = Depends(get_current_session)):
    """Get current project information"""
    project = session["project"]
    return ProjectInfoResponse(
        name=project.name,
        has_design=project.design_matrix is not None,
        has_results=project.results_data is not None,
        factors_count=len(project.get_factors()),
        design_runs=len(project.design_matrix) if project.design_matrix is not None else None,
    )


@router.put("/name")
async def update_project_name(
    body: dict,
    session: dict = Depends(get_current_session),
):
    """Update project name"""
    session["project"].name = body.get("name", "Untitled Project")
    return {"success": True}


@router.get("/save")
async def save_project(session: dict = Depends(get_current_session)):
    """Download project as JSON file"""
    project = session["project"]

    # Save to temporary file then read
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
        project.save(tmp_path)

    with open(tmp_path, "rb") as f:
        content = f.read()
    os.unlink(tmp_path)

    filename = f"{project.name}.json"
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/load")
async def load_project(
    file: UploadFile = File(...),
    session: dict = Depends(get_current_session),
):
    """Upload and load a project JSON file"""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        project = DoEProject.load(tmp_path)
        session["project"] = project
        return ProjectInfoResponse(
            name=project.name,
            has_design=project.design_matrix is not None,
            has_results=project.results_data is not None,
            factors_count=len(project.get_factors()),
            design_runs=len(project.design_matrix) if project.design_matrix is not None else None,
        )
    finally:
        os.unlink(tmp_path)
