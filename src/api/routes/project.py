from pathlib import Path
from fastapi import APIRouter, HTTPException

from api.model.project import ProjectCreateBody, ProjectListFileBody, ProjectOpenBody, ProjectOpenFileBody, ProjectSaveBody, ProjectTabBody, ProjectTabIOBody
from axium.project import AxiumProjectManager, AxiumProjectStateSave

router = APIRouter(prefix="/project", tags=["Project"])

@router.post("")
def create_project(body: ProjectCreateBody):
    try:
        project_dir = AxiumProjectManager.create_project(body.path, body.name)
        return {"project_dir": project_dir}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("")
def save_project(body: ProjectSaveBody):
    try:
        project = AxiumProjectManager.open_project(body.path)
        
        project.name  = body.name
        project.state = AxiumProjectStateSave.from_dict(body.state)
        
        AxiumProjectManager.save_project(body.path, project)
        return {"project": project.to_dict()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/open")
def open_project(body: ProjectOpenBody):
    try:
        project = AxiumProjectManager.open_project(body.path)
        return {"project": project.to_dict()}

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/recent")
def get_recent_projects():
    try:
        recent = AxiumProjectManager.get_recent_projects()
        return {"recent_projects": recent}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/file/list")
def list_file(body: ProjectListFileBody):
    return AxiumProjectManager.list_files(body.path)

@router.post("/file/open")
def open_file(body: ProjectOpenFileBody):
    path = Path(body.path)
    content = AxiumProjectManager.open_file(body.path)

    return {
        "name": path.name,
        "content": content
    }

@router.post("/tab/list")
def list_tab(body: ProjectTabBody):
    path = Path(body.project_dir)
    return [
        {
            "name": Path(dir).name,
            "path": dir
        }
        for dir in AxiumProjectManager.list_tab(path)
    ]

@router.post("/tab/open")
def open_tab(body: ProjectTabIOBody):
    return AxiumProjectManager.open_tab(
        body.project_dir,
        body.file_path
    )

@router.post("/tab/close")
def close_tab(body: ProjectTabIOBody):
    return AxiumProjectManager.close_tab(
        body.project_dir,
        body.file_path
    )
    