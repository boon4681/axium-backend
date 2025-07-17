from typing import Optional
from pydantic import BaseModel

class ProjectCreateBody(BaseModel):
    name: str
    path: str  # base directory

class ProjectOpenBody(BaseModel):
    path: str  # project folder path

class ProjectSaveBody(BaseModel):
    path: str  # project folder path
    name: str
    state: Optional[dict] = None

class ProjectListFileBody(BaseModel):
    path: str  # project folder path