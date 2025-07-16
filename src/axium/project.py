import os
import json
from datetime import datetime
from pathlib import Path
from typing import Union
import platform

class AxiumProjectManager:
    @staticmethod
    def create_project(base_dir: Union[str, Path], project_name: str):
        """
        Create a new project folder in the user-selected base_dir, and place a .axium file inside.
        Returns the path to the created project folder.
        """
        base_dir = Path(base_dir)
        project_dir = base_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        project_file = project_dir / f"{project_name}.axium"
        project = AxiumProjectFile(name=project_name)
        
        with open(project_file, 'w') as f:
            json.dump(project.to_dict(), f, indent=2)

        return str(project_dir)

    @staticmethod
    def open_project(project_dir: Union[str, Path]):
        """
        Open a project by folder path. Loads the .axium file inside the folder.
        """
        
        project_dir = Path(project_dir)
        axium_files = list(project_dir.glob("*.axium"))
        
        if not axium_files:
            raise FileNotFoundError(f"No .axium file found in {project_dir}")
        with open(axium_files[0], 'r') as f:
            data = json.load(f)
        
        return AxiumProjectFile.from_dict(data)

    @staticmethod
    def save_project(project_dir: Union[str, Path], project_file_obj):
        """
        Save the project file object to the .axium file in the given folder.
        """
        project_dir = Path(project_dir)
        axium_files = list(project_dir.glob("*.axium"))
        
        if not axium_files:
            # If no .axium file, create one with the project name
            axium_path = project_dir / f"{project_file_obj.name}.axium"
        else:
            axium_path = axium_files[0]
        
        project_file_obj.last_modified = datetime.now().isoformat()
        with open(axium_path, 'w') as f:
            json.dump(project_file_obj.to_dict(), f, indent=2)

    @staticmethod
    def get_recent_projects():
        """
        Get the list of recent projects from the backend config file.
        Returns a list of project folder paths (as strings).
        """
        # Use a user config directory, e.g., ~/.axium/recent_projects.json
        if platform.system() == "Windows":
            config_dir = Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'axium'
        else:
            config_dir = Path.home() / '.axium'

        config_dir.mkdir(exist_ok=True)
        recent_file = config_dir / 'recent_projects.json'

        if not recent_file.exists():
            with open(recent_file, 'w') as f:
                json.dump([], f)
            return []

        with open(recent_file, 'r') as f:
            try:
                recent = json.load(f)
                if not isinstance(recent, list):
                    return []
                return recent
            except Exception:
                return []

    @staticmethod
    def add_recent_project(project_dir: Union[str, Path]):
        """
        Add a project to the recent projects list.
        """
        recent = AxiumProjectManager.get_recent_projects()
        recent.insert(0, project_dir)

class AxiumProjectFile:
    def __init__(self, name, created_date=None, last_modified=None, state=None):
        self.name = name
        self.created_date = created_date or datetime.now().isoformat()
        self.last_modified = last_modified or datetime.now().isoformat()
        self.state = state or AxiumProjectStateSave()

    def to_dict(self):
        return {
            "name": self.name,
            "created_date": self.created_date,
            "last_modified": self.last_modified,
            "state": self.state.to_dict() if self.state else {},
        }

    @classmethod
    def from_dict(cls, data):
        state = AxiumProjectStateSave.from_dict(data.get("state", {}))
        return cls(
            name=data["name"],
            created_date=data.get("created_date"),
            last_modified=data.get("last_modified"),
            state=state,
        )

class AxiumProjectStateSave:
    def __init__(self, nodes=[], edges=[], extra={}):
        self.nodes = nodes
        self.edges = edges # List of edge dicts
        self.extra = extra # Any additional state

    def to_dict(self):
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            nodes=data.get("nodes"),
            edges=data.get("edges"),
            extra=data.get("extra"),
        )