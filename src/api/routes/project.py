from pathlib import Path
from fastapi import APIRouter, HTTPException

from api.model.project import ProjectCreateBody, ProjectListFileBody, ProjectOpenBody, ProjectOpenFileBody, ProjectSaveBody, ProjectSaveFileBody, ProjectTabBody, ProjectTabIOBody
from axium.project import AxiumProjectManager, AxiumProjectStateSave

router = APIRouter(prefix="/project", tags=["Project"])

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

@router.post("/file/save")
def patch_or_create_file(body: ProjectSaveFileBody):
    path = Path(body.path)
    content = AxiumProjectManager.save_file(path, body.content)
    
    return {
        "name": path.name,
        "content": content
    }

@router.post("/tab/list")
def list_tab(body: ProjectTabBody):
    path = Path(body.project_dir)
    return {
        "project_path": path.absolute(),
        "project_tab": [
            {
                "name": Path(dir).name,
                "path": dir
            }
            for dir in AxiumProjectManager.list_tab(path)
        ]
    }
    

@router.post("/tab/open")
def open_tab(body: ProjectTabIOBody):
    return {
        "project_path": Path(body.project_dir).absolute(),
        "project_tab": [
            {
                "name": Path(dir).name,
                "path": dir
            }
            for dir in AxiumProjectManager.open_tab(
                body.project_dir,
                body.file_path
            )
        ]
    }

@router.post("/tab/close")
def close_tab(body: ProjectTabIOBody):
    return {
        "project_path": Path(body.project_dir).absolute(),
        "project_tab": [
            {
                "name": Path(dir).name,
                "path": dir
            }
            for dir in AxiumProjectManager.close_tab(
                body.project_dir,
                body.file_path
            )
        ]
    }
    