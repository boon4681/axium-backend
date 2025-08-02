from typing import List, Dict, Union

File = Dict[str, str]
Folder = Dict[str, Union[str, List['FileNode']]]
FileNode = Union[File, Folder]
FileTree = List[FileNode]

