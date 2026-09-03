import os
import queue
import time
import torch
import signal
import typing as T
from types import SimpleNamespace
from abc import ABC, abstractmethod

import torch.multiprocessing as mp
from multiprocessing.context import SpawnProcess

from Module.Map import TensorBundle, VisualMap
from Utility.PrettyPrint import Logger
from Utility.Extensions import ConfigTestableSubclass
from Utility.Timer import Timer

if T.TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = object


T_GraphInput  = T.TypeVar("T_GraphInput", bound=DataclassInstance)
T_Context     = T.TypeVar("T_Context")
T_GraphOutput = T.TypeVar("T_GraphOutput", bound=DataclassInstance)


class IOptimizer(ABC, T.Generic[T_GraphInput, T_Context, T_GraphOutput], ConfigTestableSubclass):
    """
    Interface for optimization module. When config.parallel set to `true`, will spawn a child process
    to run optimization loop in "background".

    `IOptimizer.optimize(global_map: TensorMap, frames: BatchFrames) -> None`

    * In sequential mode, will run optimization loop in blocking mannor and return when optimization is finished.

    * In parallel mode, will send optimization job to child process and return immediately (non-blocking).

    `IOptimizer.write_back(global_map: TensorMap) -> None`

    * In sequential mode, will write back optimization result to global_map immediately and return.

    * In parallel mode, will wait for child process to finish optimization job and write back result to global_map. (blocking)

    `IOptimizer.terminate() -> None`

    Force terminate child process if in parallel mode. no-op if in sequential mode.
    """
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()

        self.config: SimpleNamespace = config
        self.is_parallel_mode = config.parallel

        # For sequential mode
        self.context     : None | T_Context  = None
        self.optimize_res: None | T_GraphOutput = None

        # For parallel mode
        self.send_queue: None | mp.Queue = None
        self.recv_queue: None | mp.Queue = None
        self.child_proc: None | SpawnProcess = None   # child process running optimization loop

        # Keep track if there is a job on child to avoid deadlock (wait forever for child to finish
        # a non-exist job)
        self.has_opt_job: bool = False
        self.parallel_timeout_s: float | None = getattr(config, 'parallel_timeout_s', None)

        if self.is_parallel_mode:
            ctx = mp.get_context("spawn")

            self.send_queue = ctx.Queue() # Parent -> Child
            self.recv_queue = ctx.Queue() # Child -> Parent
            # Spawn child process
            self.child_proc = ctx.Process(
                target=IOptimizerParallelWorker,
                args=(config, self.init_context, self._optimize, self.send_queue, self.recv_queue),
            )
            assert self.child_proc is not None
            self.child_proc.start()

        if not self.is_parallel_mode: # This is called by parent process
            self.context = self.init_context(config)

    ### Internal Interface to be implemented by the user
    @staticmethod
    @abstractmethod
    def init_context(config) -> T_Context:
        """
        Given config, initialize a *mutable* context object that is preserved between optimizations.

        Can also be used to avoid repetitive initialization of some objects (e.g. optimizer, robust kernel).
        """
        ...

    @staticmethod
    @abstractmethod
    def _optimize(context: T_Context, graph_data: T_GraphInput) -> tuple[T_Context, T_GraphOutput]:
        """
        Given context and argument, construct the optimization problem, solve it and return the
        updated context and result.
        """
        ...

    @abstractmethod
    def get_graph_data(self, global_map: VisualMap, frame_idx: torch.Tensor,
                       observations: torch.Tensor | None = None, edges: torch.Tensor | None = None) -> T_GraphInput: ...

    @abstractmethod
    def write_graph_data(self, result: T_GraphOutput | None, global_map: VisualMap) -> None: ...

    ### Implementation detail

    ## Sequential Version   #############################################################
    def __launch_optim_sequential(self, graph_data: T_GraphInput) -> None:
        assert self.context is not None
        self.has_opt_job = True
        with Timer.CPUTimingContext("Optimizer.optimize"):
            self.context, self.optimize_res = self._optimize(self.context, graph_data)

    def __get_output_sequential(self) -> T_GraphOutput | None:
        return self.optimize_res

    ## Parallel Version
    def __launch_optim_parallel(self, graph_data: T_GraphInput) -> None:
        assert self.send_queue is not None
        assert self.child_proc and self.child_proc.is_alive()
        self.send_queue.put(graph_data)
        self.has_opt_job = True

    def __get_output_parallel(self, timeout_s: float | None = None) -> T_GraphOutput | None:
        assert self.recv_queue is not None
        assert self.child_proc and self.child_proc.is_alive()
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        while self.has_opt_job:
            try:
                result = self.recv_queue.get(timeout=0.1)
                self.has_opt_job = False
                return result
            except queue.Empty:
                if not self.child_proc.is_alive():
                    raise RuntimeError("Optimizer child process exited unexpectedly!")
                if deadline is not None and time.monotonic() > deadline:
                    Logger.write("warn", "Timeout while waiting for optimizer result!")
                    self.has_opt_job = False # abandon this job
                    return None
        return None

    ### External Interface used by Users    #############################################
    @T.final
    @property
    def is_running(self) -> bool:
        """
        Returns immediately, indicate the status of optimizer:
        - true if there is a job on child process running.
        - false otherwise - which includes the following cases:
            1. The optimizer is not running in parallel mode (no child process)
            2. The job on child process is finished.
            3. The child process have not received optimization job yet.
        """
        if self.send_queue is None: return False         # Not in Parallel Mode
        if not self.has_opt_job: return False      # No Job on Child
        return self.recv_queue.empty()    # Job on Child but not finished

    @T.final
    @property
    def InputType(self) -> type[T_GraphInput]:
        """
        Returns the concrete type used for T_GraphInput. Raises TypeError if not explicitly provided.
        """
        orig_bases = getattr(self.__class__, "__orig_bases__", [])
        for base in orig_bases:
            if hasattr(base, "__args__") and len(base.__args__) >= 1:
                input_type = base.__args__[0]
                if input_type is not T.TypeVar and not isinstance(input_type, T.TypeVar):
                    return input_type
        raise TypeError("T_GraphInput not explicitly specified in IOptimizer subclass.")

    @T.final
    @property
    def OutputType(self) -> type[T_GraphOutput]:
        """
        Returns the concrete type used for T_GraphOutput. Raises TypeError if not explicitly provided.
        """
        orig_bases = getattr(self.__class__, "__orig_bases__", [])
        for base in orig_bases:
            if hasattr(base, "__args__") and len(base.__args__) >= 3:
                output_type = base.__args__[2]
                if output_type is not T.TypeVar and not isinstance(output_type, T.TypeVar):
                    return output_type
        raise TypeError("T_GraphOutput not explicitly specified in IOptimizer subclass.")

    def get_optimal(self) -> T_GraphOutput | None:
        if self.is_parallel_mode:
            return self.__get_output_parallel()
        else:
            return self.__get_output_sequential()

    def start_optimize(self, graph_data: T_GraphInput) -> None:
        if self.is_parallel_mode:
            # Send T_GraphArg to child process using Queue
            self.__launch_optim_parallel(graph_data)
        else:
            self.__launch_optim_sequential(graph_data)

    def sequential_optimize(self, graph_data: T_GraphInput) -> T_GraphOutput:
        assert self.context is not None
        self.context, optim_res = self._optimize(self.context, graph_data)
        return optim_res

    def write_map(self, global_map: VisualMap):
        if self.is_parallel_mode:
            graph_res_local = self.__get_output_parallel(self.parallel_timeout_s)
        else:
            graph_res_local = self.__get_output_sequential()

        self.write_graph_data(graph_res_local, global_map)

    def finalize(self, global_map: VisualMap) -> None:
        """Optional end-of-sequence pass over the whole map, called once by the
        odometry's terminate() after the last write_map() and before terminate().
        A backend that keeps state across frames (e.g. a persistent graph) can
        re-optimize offline here and write smoothed poses back. Default: no-op."""
        return None

    def terminate(self):
        if self.child_proc and self.child_proc.is_alive():
            self.child_proc.terminate()

    def frame_stats(self) -> dict[str, T.Any] | None:
        """Optional per-frame backend statistics (column arrays keyed by name),
        saved by MACVO.py as optimizer_stats.npz. Default: none."""
        return None


def IOptimizerParallelWorker(
    config: SimpleNamespace,
    init_context: T.Callable[[SimpleNamespace,], T_Context],
    optimize: T.Callable[[T_Context, T_GraphInput], tuple[T_Context, T_GraphOutput]],
    send_queue: mp.Queue,
    recv_queue: mp.Queue
):
    Logger.write("info", "OptimizationParallelWorker started")
    # NOTE: child process ignore keyboard interrupt to ignore deadlock
    # (parent process will terminate child process on exit in MAC-VO implementation)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    context = init_context(config) # This is called by child process
    while True:
        graph_args: T_GraphInput = send_queue.get()  # blocks until work arrives, no spin
        graph_args_local = graph_args

        context, graph_res = optimize(context, graph_args_local)
        recv_queue.put(graph_res)
