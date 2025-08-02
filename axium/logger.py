from collections import deque
from datetime import datetime
import io
import logging as _logging
import sys
import threading
import asyncio
import functools
global print
print = functools.partial(print, flush=True)
logging = _logging.getLogger()

logs: list = None
stdout_intcp = None
stderr_intcp = None


class Interceptor(io.TextIOWrapper):
    flush_listeners: list = []

    def __init__(self, stream, *args, **kwargs):
        buffer = stream.buffer
        encoding = stream.encoding
        super().__init__(
            buffer,
            *args,
            **kwargs,
            encoding=encoding,
            line_buffering=stream.line_buffering
        )
        self.flush_listeners = []
        self.lock = threading.Lock()
        self.logs_buffer: list[str] = []

    def write(self, data):
        entry = {"t": datetime.now().isoformat(), "m": data}
        with self.lock:
            self.logs_buffer.append(entry)
            if isinstance(data, str) and data.startswith("\r") and not logs[-1]["m"].endswith("\n"):
                logs.pop()
            logs.append(entry)
        super().write(data)

    def flush(self):
        super().flush()
        with self.lock:
            if not self.logs_buffer:
                return
            logs_to_flush = self.logs_buffer
            self.logs_buffer = []
        for cb in self.flush_listeners:
            cb(logs_to_flush)

    def add_event_listener(self, event: str, callback):
        if event == "flush":
            self.flush_listeners.append(callback)
            return
        raise RuntimeError(f"Unknown event named \"{event}\"")


def get_logs():
    return logs


def add_event_listener(event: str, callback):
    if stdout_intcp is not None:
        stdout_intcp.add_event_listener(event, callback)
    if stderr_intcp is not None:
        stderr_intcp.add_event_listener(event, callback)


def setup_logger(log_level: str = "INFO", capacity: int = 300, use_stdout: bool = False):
    global logs
    if logs:
        return
    logs = deque(maxlen=capacity)

    global stdout_intcp
    global stderr_intcp

    stdout_intcp = sys.stdout = Interceptor(sys.stdout)
    stderr_intcp = sys.stderr = Interceptor(sys.stderr)

    logging.setLevel(log_level)

    stream_handler = _logging.StreamHandler()
    stream_handler.setFormatter(_logging.Formatter("%(message)s"))

    if use_stdout:
        stream_handler.addFilter(
            lambda record:
            not record.levelno < _logging.ERROR
        )

        stdout_handler = _logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(_logging.Formatter("%(message)s"))
        stdout_handler.addFilter(
            lambda record:
            record.levelno < _logging.ERROR
        )
        logging.addHandler(stdout_handler)

    logging.addHandler(stream_handler)
