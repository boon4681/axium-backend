from typing import Optional, Dict
from pydantic import BaseModel

class ProjectCreateBody(BaseModel):
    name: str
    path: str  # base directory

class ProjectOpenBody(BaseModel):
    path: str  # project folder path

class ProjectSaveBody(BaseModel):
    path: str  # project folder path
    name: str
    state: Optional[Dict] = None

class ProjectListFileBody(BaseModel):
    path: str  # project folder path

class ProjectOpenFileBody(BaseModel):
    path: str  # project file path

class ProjectSaveFileBody(BaseModel):
    path: str  # project file path
    content: str

class ProjectTabBody(BaseModel):
    project_dir: str  # project file path

class ProjectTabIOBody(BaseModel):
    project_dir: str  # project file path
    file_path: str
