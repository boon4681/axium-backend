import sys
import traceback
from pydantic import TypeAdapter, ValidationError
from watchdog.events import FileSystemEventHandler
import axium.logger
import logging as _logging
from multiprocessing import Queue as MQueue
import time
import os
from pathlib import Path
from watchdog.observers import Observer
from axium.axium_typing import *
import axium.utils as utils
from hashlib import blake2b
import json

from axium.validator import AxiumErrorJSONResponse, AxiumProjectMetaData


class WatchProject(FileSystemEventHandler):
    listeners: list = []
    files: FileTree = []

    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.files = utils.list_files_tree(root)

    def add_event_listener(self, event: str, callback):
        if event == "*":
            self.listeners.append(callback)
            return
        raise RuntimeError(f"Unknown event named \"{event}\"")

    def flush(self):
        self.files = utils.list_files_tree(self.root)
        for cb in self.listeners:
            cb(self.files)

    def on_created(self, event):
        # if event.is_directory:
        #     print(f"Directory created: {event.src_path}")
        # else:
        #     print(f"File created: {event.src_path}")
        self.flush()

    def on_deleted(self, event):
        # if event.is_directory:
        #     print(f"Directory deleted: {event.src_path}")
        # else:
        #     print(f"File deleted: {event.src_path}")
        self.flush()

    def on_modified(self, event):
        # print(Path(event.src_path).as_posix() in self.files)
        # if event.is_directory:
        #     print(f"Directory modified: {event.src_path}")
        # else:
        #     print(f"File modified: {event.src_path}")
        self.flush()

    def on_moved(self, event):
        # if event.is_directory:
        #     print(
        #         f"Directory moved from {event.src_path} to {event.dest_path}")
        # else:
        #     print(f"File moved from {event.src_path} to {event.dest_path}")
        self.flush()


class AxiumProject:
    current_tab: str = ""
    tabs: list[str] = []

    def __init__(self, lock, queue: MQueue, path: str):
        self.queue = queue
        self.path = path
        self.lock = lock
        self.root = Path(path)
        axium.logger.setup_logger(log_level="INFO", use_stdout=True)
        axium.logger.add_event_listener(
            "flush", lambda x: self.emit("logs", x)
        )
        self.emit("ready", self.load_project_meta())
        axium.logger.logging.info("HI")
        self.logging = axium.logger.logging
        self.run_command("list-files", None)
        # self.run_command("save-project-meta", {})
        event_handler = WatchProject(path)
        event_handler.add_event_listener(
            "*", lambda x: self.run_command("list-files", None)
        )
        observer = Observer()
        observer.schedule(event_handler, self.path, recursive=True)
        observer.start()
        self.project_observer = observer

    def _run_command(self, command: str, data):
        with self.lock:
            try:
                if command == "list-files":
                    return self.list_files()
                if command == "save-project-meta":
                    return self.save_project_meta(data)
                if command == "write-file":
                    return self.write_file(data['path'], data['content'])
                if command == "read-file":
                    return self.read_file(data['path'])
            except Exception as e:
                self.logging.error("\n".join(traceback.format_exception(e)))
                exc_info = sys.exc_info()
                sExceptionInfo = ''.join(traceback.format_exception(*exc_info))
                exc_type, exc_value, exc_context = sys.exc_info()

                return utils.AxiumErrorJSON(
                    {
                        "status": "error",
                        "type": exc_type.__name__,
                        "description": str(exc_value),
                        "details": sExceptionInfo
                    }
                )

    def run_command(self, command: str, data):
        result = self._run_command(command, data)
        if isinstance(result, AxiumErrorJSONResponse):
            self.emit("res.error", result.model_dump())
            return
        self.emit("res."+command, result)

    def list_files(self):
        return utils.list_files_tree(self.path)

    def get_meta_dir(self):
        project_root = Path(self.path)
        h = blake2b(digest_size=12)
        h.update(self.path.encode("UTF-8"))
        digest = h.hexdigest()
        meta_dir = "-".join(
            [project_root.parent.name, project_root.name, digest])
        return meta_dir

    def load_project_meta(self):
        root = utils.HOME_DIR() / self.get_meta_dir()
        if not root.exists():
            self.save_project_meta()
        with open(root / "project.json", "r") as fs:
            data = json.load(fs)
            v = TypeAdapter(AxiumProjectMetaData)
            try:
                return v.validate_python(data).model_dump()
            except ValidationError as e:
                self.logging.error(e)
                return None

    def save_project_meta(self, data):
        root = utils.HOME_DIR() / self.get_meta_dir()
        if not root.exists():
            os.mkdir(root)
        v = TypeAdapter(AxiumProjectMetaData)
        metadata = None
        try:
            metadata = v.validate_python(data).model_dump()
        except ValidationError as e:
            self.logging.error(e)
            return False
        with open(root / "project.json", "w") as fs:
            fs.write(json.dumps(metadata))
        return True

    def write_file(self, path: str, content: str):
        dir = Path(path)
        if not dir.as_posix().startswith(self.root.as_posix()):
            raise RuntimeError(
                f"Cannot write file outside project root directory at {path}")
        with open(path, "w") as fs:
            fs.write(content)
        return True

    def read_file(self, path: str):
        dir = Path(path)
        if dir.is_file():
            with open(dir, "r") as fs:
                return {
                    "name": str(dir.name),
                    "path": path,
                    "content": fs.read()
                }
            return None
        return None

    def emit(self, event: str, data=None, room: str = None):
        self.queue.put(
            {
                "event": event,
                "data": data,
                "sid": None,
                "room": room if room is not None else self.path
            }
        )

    def close(self):
        self.project_observer.stop()
        self.project_observer.join()
