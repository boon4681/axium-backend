import os
import json
from datetime import datetime
from pathlib import Path
import platform

class AxiumProjectManager:

    @staticmethod
    def create_project(base_dir: str | Path, project_name: str):
        """
        Create a new project folder in the user-selected base_dir, and place a .axm file inside.
        Returns the path to the created project folder.
        """
        base_dir = Path(base_dir)
        project_dir = base_dir / project_name

        if project_dir.exists():
            raise FileExistsError(f"Project name {project_name} is already exist")

        project_dir.mkdir(parents=True, exist_ok=True)

        project_file = project_dir / f"{project_name}.axm"
        project = AxiumProjectFile(name=project_name)
        
        with open(project_file, 'w') as f:
            json.dump(project.to_dict(), f, indent=2)

        AxiumProjectManager.add_recent_project(str(project_dir))

        return str(project_dir)

    @staticmethod
    def open_project(project_dir: str | Path):
        """
        Open a project by folder path. Loads the .axm file inside the folder.
        """
        
        project_dir = Path(project_dir)
        if not project_dir.exists():
            raise FileNotFoundError("Project directory is not found")

        axium_files = list(project_dir.glob("*.axm"))
        if not axium_files:
            raise FileNotFoundError(f"No .axm file found in {project_dir}")
        
        with open(axium_files[0], 'r') as file:
            data = json.load(file)
        
        return AxiumProjectFile.from_dict(data)

    @staticmethod
    def save_project(project_dir: str | Path, project_file_obj):
        """
        Save the project file object to the .axm file in the given folder.
        """
        project_dir = Path(project_dir)
        axium_files = list(project_dir.glob("*.axm"))
        
        if not axium_files:
            # If no .axm file, create one with the project name
            axium_path = project_dir / f"{project_file_obj.name}.axm"
        else:
            axium_path = axium_files[0]
        
        project_file_obj.last_modified = datetime.now().isoformat()

        with open(axium_path, 'w') as f:
            json.dump(project_file_obj.to_dict(), f, indent=2)

    @staticmethod
    def read_recent_projects_file():
        if platform.system() == "Windows":
            config_dir = Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'axium'
        else:
            config_dir = Path.home() / '.axm'

        recent_file = config_dir / 'recent_projects.json'
        if not config_dir.exists():
            os.mkdir(config_dir)

        if not recent_file.exists():
            with open(recent_file, 'w') as f:
                json.dump([], f)
        
        return recent_file

    @staticmethod
    def get_recent_projects():
        """
        Get the list of recent projects from the backend config file.
        Returns a list of project folder paths (as strings).
        """

        recent_file = AxiumProjectManager.read_recent_projects_file()
        with open(recent_file, 'r') as f:
            try:
                recent = json.load(f)
                return [path for path in recent if Path(path).exists()]
            
            except Exception:
                return []

    @staticmethod
    def add_recent_project(project_dir: str | Path):
        """
        Add a project to the recent projects list.
        """

        recent_project = AxiumProjectManager.get_recent_projects()
        recent_project = list(set([*recent_project, project_dir]))
        
        recent_file = AxiumProjectManager.read_recent_projects_file()
        with open(recent_file, 'w') as f:
            json.dump(recent_project, f)

        return AxiumProjectManager.get_recent_projects()
    
    @staticmethod
    def list_files(project_dir: str | Path):
        """
            Recursively call file and folder in project directory
        """

        project_dir = Path(project_dir)
        if not project_dir.exists():
            raise FileNotFoundError("Project directory is not found")

        def walk_dir(current_path: Path):
            file_tree = []

            # Sort by folder-order-first then sort by name
            for entry in sorted(current_path.iterdir(), key=lambda x: (x.is_file(), x.name)):
                if entry.is_file():
                    file_tree.append({
                        "type": "file",
                        "path": entry.absolute(),
                        "name": entry.name
                    })
                elif entry.is_dir():
                    file_tree.append({
                        "type":  "folder",
                        "path":  entry.absolute(),
                        "name":  entry.name,
                        "files": walk_dir(entry)
                    })

            return file_tree

        return walk_dir(project_dir)

    @staticmethod
    def open_file(file_dir: str | Path):
        with open(file_dir, 'r') as file:
            return file.read()
        
    @staticmethod
    def save_file(file_dir: str | Path, content: str):
        file_path = Path(file_dir)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(file_path)
        
    @staticmethod
    def read_list_tabs():
        if platform.system() == "Windows":
            config_dir = Path(os.getenv('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'axium'
        else:
            config_dir = Path.home() / '.axm'

        if not config_dir.exists():
            os.mkdir(config_dir)

        project_tabs = config_dir / 'project_tabs.json'

        if not project_tabs.exists():
            with open(project_tabs, 'w') as f:
                json.dump({}, f)
        
        return project_tabs

    @staticmethod
    def list_tab(project_dir: str | Path):
        list_tab_path = AxiumProjectManager.read_list_tabs()

        with open(list_tab_path, 'r') as f:
            obj = json.load(f)
            if project_dir in obj:
                return obj[str(project_dir)]
            else:
                obj[str(project_dir)] = []
            with open(list_tab_path, 'w') as fs:
                fs.write(json.dumps(obj))

        return []

    @staticmethod
    def open_tab(project_dir: str | Path, path: str):
        recent_tab = AxiumProjectManager.list_tab(project_dir)

        if path not in recent_tab:
            recent_tab.append(path)

        list_tab_path = AxiumProjectManager.read_list_tabs()
        with open(list_tab_path, "r") as f:
            obj = json.load(f)

        obj[project_dir] = recent_tab
        with open(list_tab_path, "w") as f:
            json.dump(obj, f, indent=2)
        
        return AxiumProjectManager.list_tab(project_dir)
    
    @staticmethod
    def close_tab(project_dir: str | Path, path: str):
        recent_tab = AxiumProjectManager.list_tab(project_dir)

        if path in recent_tab:
            recent_tab.remove(path)

        list_tab_path = AxiumProjectManager.read_list_tabs()
        with open(list_tab_path, "r") as f:
            obj = json.load(f)

        obj[project_dir] = recent_tab
        with open(list_tab_path, "w") as f:
            json.dump(obj, f, indent=2)
        
        return AxiumProjectManager.list_tab(project_dir)
    
class AxiumProjectFile:
    def __init__(self, name, created_date=None, last_modified=None, state=None):
        self.name = name
        self.created_date = created_date or datetime.now().isoformat()
        self.last_modified = last_modified or datetime.now().isoformat()
        self.state = state or AxiumProjectStateSave()

        self.file = []

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
            nodes=data.get("nodes") or [],
            edges=data.get("edges") or [],
            extra=data.get("extra") or [],
        )    