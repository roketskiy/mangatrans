"""Windows process-tree accounting, including children that have already exited."""
import ctypes as C
from ctypes import wintypes as W
import subprocess

import psutil


class IO(C.Structure):
    _fields_ = [(n, C.c_ulonglong) for n in (
        "read_operations", "write_operations", "other_operations",
        "read_bytes", "write_bytes", "other_bytes")]


class Basic(C.Structure):
    _fields_ = [(n, C.c_longlong) for n in (
        "user_time", "kernel_time", "period_user_time", "period_kernel_time")]
    _fields_ += [(n, W.DWORD) for n in (
        "page_faults", "total_processes", "active_processes", "terminated_processes")]


class Accounting(C.Structure):
    _fields_ = [("basic", Basic), ("io", IO)]


class Limits(C.Structure):
    _fields_ = [("process_time", C.c_longlong), ("job_time", C.c_longlong),
                ("flags", W.DWORD), ("min_working_set", C.c_size_t),
                ("max_working_set", C.c_size_t), ("active_limit", W.DWORD),
                ("affinity", C.c_size_t), ("priority", W.DWORD),
                ("scheduling", W.DWORD)]


class Extended(C.Structure):
    _fields_ = [("basic", Limits), ("io", IO),
                ("process_memory_limit", C.c_size_t), ("job_memory_limit", C.c_size_t),
                ("peak_process_commit", C.c_size_t), ("peak_job_commit", C.c_size_t)]


class Job:
    def __init__(self):
        self.api = C.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([C.c_void_p, W.LPCWSTR], W.HANDLE),
            "SetInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
            "QueryInformationJobObject": ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL),
            "AssignProcessToJobObject": ([W.HANDLE, W.HANDLE], W.BOOL),
            "OpenThread": ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            "ResumeThread": ([W.HANDLE], W.DWORD),
            "CloseHandle": ([W.HANDLE], W.BOOL),
            "TerminateJobObject": ([W.HANDLE, W.UINT], W.BOOL),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.api, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        self.check(self.handle)
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self.check(self.api.SetInformationJobObject(
                self.handle, 9, C.byref(limits), C.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(value):
        if not value:
            raise C.WinError(C.get_last_error())
        return value

    def launch(self, command, log, cwd):
        # The root cannot spawn any children before assignment to the job.
        proc = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=log,
                                stdin=subprocess.DEVNULL,
                                creationflags=subprocess.CREATE_NO_WINDOW | 0x4)
        try:
            self.check(self.api.AssignProcessToJobObject(self.handle, int(proc._handle)))
            threads = psutil.Process(proc.pid).threads()
            if len(threads) != 1:
                raise RuntimeError("Suspended process did not have exactly one initial thread")
            thread = self.check(self.api.OpenThread(0x2, False, threads[0].id))
            try:
                if self.api.ResumeThread(thread) == 0xFFFFFFFF:
                    raise C.WinError(C.get_last_error())
            finally:
                self.api.CloseHandle(thread)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        return proc

    def totals(self):
        a, m = Accounting(), Extended()
        self.check(self.api.QueryInformationJobObject(
            self.handle, 8, C.byref(a), C.sizeof(a), None))
        self.check(self.api.QueryInformationJobObject(
            self.handle, 9, C.byref(m), C.sizeof(m), None))
        return dict(cpu_user_s=a.basic.user_time / 1e7,
                    cpu_kernel_s=a.basic.kernel_time / 1e7,
                    cpu_s=(a.basic.user_time + a.basic.kernel_time) / 1e7,
                    page_faults=a.basic.page_faults,
                    total_processes=a.basic.total_processes,
                    active_processes=a.basic.active_processes,
                    peak_job_commit_bytes=m.peak_job_commit,
                    peak_process_commit_bytes=m.peak_process_commit,
                    **{name: getattr(a.io, name) for name, _ in IO._fields_})

    def terminate(self):
        self.check(self.api.TerminateJobObject(self.handle, 124))

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
