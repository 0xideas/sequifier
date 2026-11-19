"""Opt-in spawned simulators with explicit requests, deadlines and bounded cleanup."""

import multiprocessing
import pickle
import time
import traceback
from contextlib import contextmanager
from multiprocessing.reduction import ForkingPickler

from sequifier.io.memory_budget import resident_bytes
from sequifier.rl.environment import (
    adapter_identity,
    load_online_adapter,
    require_group_start,
)
from sequifier.runtime.resource_accounting import allocation


def require_parallel_group_start(adapter):
    require_group_start(adapter)
    try:
        supported = adapter.supports_parallel_group_start
    except AttributeError:
        supported = False
    if supported is not True:
        raise ValueError(
            "parallel GRPO requires adapter.supports_parallel_group_start = True; "
            "its portable group-start tokens must work in independent adapter instances"
        )


def _serve(connection, spec, threads):
    adapter = None
    request = 0
    try:
        import torch

        torch.set_num_threads(threads)
        adapter = load_online_adapter(spec)
        require_parallel_group_start(adapter)
        connection.send((request, True, adapter_identity(adapter)))
        operations = {"reset_group": adapter.reset_group, "step": adapter.step}
        while True:
            request, method, args = connection.recv()
            if method == "close":
                break
            if method not in operations:
                raise ValueError(f"unsupported simulator operation: {method}")
            connection.send((request, True, operations[method](*args)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send((request, False, traceback.format_exc()[-32768:]))
        except (OSError, EOFError):
            pass
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            connection.close()


class SimulatorWorkers:
    """One in-flight request per worker; never retries an issued environment action."""

    def __init__(self, spec, count, timeout, threads, expected_identity):
        self.timeout = timeout
        self.connections, self.processes = [], []
        self.pending, self.sequence = {}, 0
        self.requests = {}
        self.closed = False
        try:
            context = multiprocessing.get_context("spawn")
            for index in range(count):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_serve,
                    args=(child, spec, threads),
                    name=f"sequifier-grpo-simulator-{index}",
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
                self.pending[index] = (0, "startup", time.monotonic() + timeout)
            for index in range(count):
                if self.receive(index) != expected_identity:
                    raise ValueError(
                        f"simulator {index} adapter identity differs from parent"
                    )
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.connections)

    def send(self, index, method, *args):
        if self.closed or index in self.pending:
            raise RuntimeError(
                "simulator is closed or already has an in-flight request"
            )
        if method not in {"reset_group", "step"}:
            raise ValueError(f"unsupported simulator operation: {method}")
        self.sequence += 1
        self.pending[index] = (self.sequence, method, time.monotonic() + self.timeout)
        try:
            payload = ForkingPickler.dumps((self.sequence, method, args))
            storage = self.requests[index] = allocation("worker_payloads")
            storage.resize(len(payload))
            self.connections[index].send_bytes(payload)
        except (OSError, EOFError) as exc:
            raise RuntimeError(
                f"simulator {index} failed while sending {method}"
            ) from exc

    def receive(self, index):
        with self.response(index) as result:
            return result

    @contextmanager
    def response(self, index):
        if self.closed or index not in self.pending:
            raise RuntimeError("simulator has no pending response")
        request, method, deadline = self.pending[index]
        connection = self.connections[index]
        with allocation("worker_payloads") as storage:
            try:
                if not connection.poll(max(0.0, deadline - time.monotonic())):
                    raise TimeoutError(f"simulator {index} timed out during {method}")
                payload = connection.recv_bytes()
                storage.resize(len(payload))
                received, ok, result = pickle.loads(payload)
            except (OSError, EOFError) as exc:
                raise RuntimeError(f"simulator {index} exited during {method}") from exc
            if received != request:
                raise RuntimeError(
                    f"simulator {index} returned an out-of-order response"
                )
            del self.pending[index]
            request_storage = self.requests.pop(index, None)
            if request_storage is not None:
                request_storage.close()
            if not ok:
                raise RuntimeError(
                    f"simulator {index} failed during {method}:\n{result}"
                )
            if storage.accounting is not None:
                decoded_size = resident_bytes(result)
                storage.resize(len(payload) + decoded_size)
                storage.resize(decoded_size)
            del payload
            yield result

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Do not queue close behind a blocked, incomplete response. Such workers
        # are terminated below; no pending collection is ever checkpointed.
        for index, connection in enumerate(self.connections):
            if index not in self.pending:
                try:
                    connection.send((-1, "close", ()))
                except (OSError, EOFError):
                    pass
        deadline = time.monotonic() + 2.0
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 2.0
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.kill()
        for process in self.processes:
            process.join(timeout=0.1)
        for connection in self.connections:
            connection.close()
        self.pending.clear()
        for storage in self.requests.values():
            storage.close()
        self.requests.clear()
