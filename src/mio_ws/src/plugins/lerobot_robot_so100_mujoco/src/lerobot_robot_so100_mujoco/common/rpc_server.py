from __future__ import annotations

import threading
from xmlrpc.server import SimpleXMLRPCServer


class JointStateRPCServer:
    def __init__(self, host: str, port: int):
        self._joint_state: dict[str, float] = {}
        self._lock = threading.Lock()
        self._server = SimpleXMLRPCServer((host, port), allow_none=True, logRequests=False)
        self._server.register_function(self.get_joint_state, "get_joint_state")
        self._server.register_function(self.ping, "ping")
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def update_joint_state(self, joint_state: dict[str, float]) -> None:
        with self._lock:
            self._joint_state = {str(name): float(value) for name, value in joint_state.items()}

    def get_joint_state(self) -> dict[str, float]:
        with self._lock:
            return dict(self._joint_state)

    def ping(self) -> bool:
        return True


__all__ = ["JointStateRPCServer"]
