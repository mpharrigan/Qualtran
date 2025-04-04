#  Copyright 2024 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import multiprocessing.connection
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import attrs
from attrs import define

if TYPE_CHECKING:
    from qualtran import Bloq


@define
class _Pending:
    """Helper dataclass to track currently executing processes in `ExecuteWithTimeout`."""

    p: multiprocessing.Process
    recv: multiprocessing.connection.Connection
    start_time: float
    kwargs: Dict[str, Any]
    task_name: str


class ExecuteWithTimeout:
    """Execute tasks in processes where each task will be killed if it exceeds `timeout`.

    Seemingly all the existing "timeout" parameters in the various built-in concurrency
    primitives in Python won't actually terminate the process. This one does.
    """

    def __init__(
        self, timeout: float, max_workers: int, *, sigterm_to_sigkill_buffer_seconds: float = 0.5
    ):
        self.timeout = timeout
        self.max_workers = max_workers
        self.sigterm_to_sigkill_buffer_seconds = sigterm_to_sigkill_buffer_seconds

        self.queued: List[Tuple[Callable, Dict[str, Any], str]] = []
        self.pending: List[_Pending] = []
        self.terminated: List[_Pending] = []

    @property
    def work_to_be_done(self) -> int:
        """The number of tasks currently executing or queued."""
        return len(self.queued) + len(self.pending)

    def submit(
        self, func: Callable, kwargs: Dict[str, Any], task_name: Optional[str] = None
    ) -> None:
        """Add a task to the queue.

        `func` must be a callable that can accept `kwargs` in addition to
        a keyword argument `cxn` which is a multiprocessing `Connection` object that forms
        the sending-half of a `mp.Pipe`. The callable must call `cxn.send(...)`
        to return a result.
        """
        if task_name is None:
            task_name = str(kwargs)
        self.queued.append((func, kwargs, task_name))

    def _submit_from_queue(self):
        # helper method that takes an item from the queue, launches a process,
        # and records it in the `pending` attribute. This must only be called
        # if we're allowed to spawn a new process.
        func, kwargs, task_name = self.queued.pop(0)
        recv, send = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=func, kwargs=kwargs | {'cxn': send})
        start_time = time.time()
        p.start()
        self.pending.append(
            _Pending(p=p, recv=recv, start_time=start_time, kwargs=kwargs, task_name=task_name)
        )

    def _scan_pendings(self) -> Optional[_Pending]:
        # helper method that goes through the currently pending tasks, terminates the ones
        # that have been going on too long, and accounts for ones that have finished.
        # Returns the `_Pending` of the killed or completed job or `None` if each pending
        # task is still running but none have exceeded the timeout.
        for i in range(len(self.pending)):
            pen = self.pending[i]

            if not pen.p.is_alive():
                self.pending.pop(i)
                pen.p.join()
                return pen

            if time.time() - pen.start_time > self.timeout:
                pen.p.terminate()
                self.pending.pop(i)
                self.terminated.append(attrs.evolve(pen, start_time=time.time()))
                return pen

        return None

    def _scan_terminated(self):
        # Go through terminated processes and make sure they actually exit.
        filtered_terminated = []
        for i in range(len(self.terminated)):
            ter = self.terminated[i]
            if not ter.p.is_alive():
                ter.p.close()
                continue

            check_time = time.time()
            if (check_time - ter.start_time) > (self.sigterm_to_sigkill_buffer_seconds):
                print(
                    f"{ter.task_name} did not respect SIGTERM after {check_time-ter.start_time}s, calling SIGKILL",
                    flush=True,
                )
                ter.p.kill()
                ter.p.join()
                ter.p.close()
                continue

            filtered_terminated.append(ter)
        self.terminated = filtered_terminated

    def n_ready_workers(self) -> int:
        return self.max_workers - len(self.pending) - len(self.terminated)

    def next_result(self) -> Tuple[Dict[str, Any], Optional[Any]]:
        """Get the next available result.

        This call is blocking, but should never take longer than `self.timeout`. This should
        be called in a loop to make sure the queue continues to be processed.

        Returns:
            task kwargs: The keyword arguments used to submit the task.
            result: If the process finished successfully, this is the object that was
                sent through the multiprocessing pipe as the result. Otherwise, the result
                is None.
        """
        while len(self.queued) > 0 and self.n_ready_workers() > 0:
            self._submit_from_queue()

        while True:
            self._scan_terminated()
            finished = self._scan_pendings()
            if finished is not None:
                break
            time.sleep(0.01)

        if finished.p.exitcode == 0:
            result = finished.recv.recv()
        else:
            result = None

        finished.recv.close()

        while len(self.queued) > 0 and self.n_ready_workers() > 0:
            self._submit_from_queue()

        return (finished.kwargs, result)


def report_on_tensors(name: str, cls_name: str, bloq: 'Bloq', cxn) -> None:
    """Get timing information for tensor functionality.

    This should be used with `ExecuteWithTimeout`. The resultant
    record dictionary is sent over `cxn`.
    """
    from qualtran.resource_counting import AtomicBloqCount, get_cost_value

    record: Dict[str, Any] = {'name': name, 'cls': cls_name}

    try:
        start = time.perf_counter()
        gc = get_cost_value(bloq, AtomicBloqCount())
        record['gates'] = gc
        record['gc_dur'] = time.perf_counter() - start

        start = time.perf_counter()
        flat = bloq.as_composite_bloq().flatten()
        record['flat_dur'] = time.perf_counter() - start
        record['flat_gc'] = len(flat.bloq_instances)

        # importing quimb is slow, so do it after flattening; but
        # don't include it in our timing info.
        from qualtran.simulation.tensor import cbloq_to_quimb

        start = time.perf_counter()
        tn = cbloq_to_quimb(flat)
        record['tn_dur'] = time.perf_counter() - start

        start = time.perf_counter()
        record['width'] = tn.contraction_width()
        record['width_dur'] = time.perf_counter() - start

    except Exception as e:  # pylint: disable=broad-exception-caught
        record['err'] = str(e)

    cxn.send(record)
