import os
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/project", tags=["Project"])

@router.get("/")
def get_last_opened_project():
    return {}

class AxiumFileBody(BaseModel):
    name: str
    path: str

@router.post("/")
def init_project(body: AxiumFileBody):

    with open(f"{body.path}/practice.cpp", "r") as file:
        print(file.read())

    return { "file_name": os.listdir(body.path) }

@router.get("/{id}")
def get_project_from_id(id: str):
    return {}

@router.patch("/{id}")
def save_project_from_id(id: str):
    return {}