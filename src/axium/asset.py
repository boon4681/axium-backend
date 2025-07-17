from typing import List, Dict, Any

class AxiumAssetFile:
    def __init__(self, id: str, name: str):
        self.type = "file"
        self.id = id
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AxiumAssetFile":
        return cls(
            id=data["id"],
            name=data["name"]
        )

class AxiumAssetFolder:
    def __init__(self, id: str, name: str, files: List["AxiumAssetFile" | "AxiumAssetFolder"]):
        self.type = "folder"
        self.id = id
        self.name = name
        self.files = files

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":  self.type,
            "id":    self.id,
            "name":  self.name,
            "files": [file.to_dict() for file in self.files]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AxiumAssetFolder":
        file_nodes = []
        
        for item in data["files"]:
            if item["type"] == "file":
                file_nodes.append(AxiumAssetFile.from_dict(item))
            elif item["type"] == "folder":
                file_nodes.append(AxiumAssetFolder.from_dict(item))

        return cls(
            id=data["id"],
            name=data["name"],
            files=file_nodes
        )


AxiumAssetFileNode = AxiumAssetFile | AxiumAssetFolder
AxiumAssetFileTree = List[AxiumAssetFileNode]