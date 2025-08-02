import asyncio
from core.ws_io import WebIO
from fastapi import FastAPI, APIRouter, WebSocket
import core.logger
from core.logger import logging
import data_test
import threading
import multiprocessing
from multiprocessing.managers import BaseManager
from axium.project import AxiumProject
from core.project import project_instances
import core.api as api
import sys
task_queue = multiprocessing.Queue()


def project_process(sid: str, path: str, conn,  lock, queue: multiprocessing.Queue):
    project = AxiumProject(lock, queue, path)

    async def flush_logs_interval():
        while True:
            await asyncio.sleep(0)
            sys.stdout.flush()
            sys.stderr.flush()

    async def event_recv():
        while True:
            await asyncio.sleep(0)
            if conn.poll():
                packet = conn.recv()
                project.run_command(packet["command"], packet["data"])

    async def main():
        await asyncio.gather(
            flush_logs_interval(),
            event_recv()
        )

    asyncio.run(main())


class AxiumServer:
    def __init__(self, app: FastAPI):
        AxiumServer.instance = self
        self.app = app
        routes = APIRouter()
        routes.include_router(api.router)
        self.routes = routes
        self.app.include_router(routes)
        sio = WebIO()
        self.sio = sio

        @sio.on('connect')
        async def connect(sid, _, meta):
            if not 'project' in meta:
                print(meta)
                await sio.disconnect(sid)
                return
            print(sid, "Connected")
            sio.enter_room(sid, meta['project'])
            parent_conn, child_conn = multiprocessing.Pipe()
            lock = multiprocessing.Lock()
            process = multiprocessing.Process(
                target=project_process,
                args=(sid, meta['project'], child_conn, lock, task_queue)
            )
            process.start()
            project_instances[sid] = {
                'process': process,
                'conn': [parent_conn, child_conn]
            }

        @sio.on('run')
        async def run(sid, data):
            print("Recv command")
            parent_conn, child_conn = project_instances[sid]['conn']
            parent_conn.send({
                "command": "run",
                "data": data
            })

        @sio.on('callable')
        async def callable(sid, data):
            parent_conn, child_conn = project_instances[sid]['conn']
            parent_conn.send({
                "command": data['command'],
                "data": data['data']
            })

        @sio.on('disconnect')
        async def disconnect(sid):
            proc_info = project_instances.pop(sid, None)

            if proc_info is not None:
                process: multiprocessing.Process = proc_info['process']
                parent_conn, child_conn = proc_info['conn']
                parent_conn.close()
                child_conn.close()
                process.terminate()
                process.join()
                print(
                    f"Terminated process {process.pid} for client {sid}"
                )

        def queue_listener(queue: multiprocessing.Queue, loop):
            while True:
                try:
                    message = queue.get()
                    print(message)
                    coro = sio.emit(
                        message['event'], message['data'], to=message['sid'], room=message['room'])
                    asyncio.run_coroutine_threadsafe(coro, loop)
                except Exception as e:
                    print(f"Error in queue listener: {e}")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        listener_thread = threading.Thread(
            target=queue_listener,
            args=(task_queue, loop),
            daemon=True
        )
        listener_thread.start()
        @app.websocket("/sio")
        async def websocket_endpoint(websocket: WebSocket):
            await sio.handle_websocket(websocket)
