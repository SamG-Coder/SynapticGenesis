"""Read-only Windows process handle used to serialize native GPU experiments."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path


class ProcessGate:
    def __init__(self, pid, expected_executable):
        if os.name != 'nt' or pid <= 0:
            raise ValueError('A positive Windows PID is required')
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        k = self.kernel
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k.GetProcessTimes.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        self.handle = k.OpenProcess(0x100000 | 0x1000, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            name, length = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
            if not k.QueryFullProcessImageNameW(self.handle, 0, name, ctypes.byref(length)):
                raise ctypes.WinError(ctypes.get_last_error())
            if Path(name.value).resolve() != Path(expected_executable).resolve():
                raise ValueError('PID executable differs from the expected native process')
            times = [wintypes.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(self.handle, *(ctypes.byref(t) for t in times)):
                raise ctypes.WinError(ctypes.get_last_error())
            created = times[0].dwLowDateTime | (times[0].dwHighDateTime << 32)
            self.identity = dict(pid=pid, executable=name.value, creation_filetime=created)
        except BaseException:
            self.close()
            raise

    def wait(self):
        # Holding the process handle also prevents a reused PID from satisfying
        # this gate. One-second waits keep the host process interruptible.
        while True:
            result = self.kernel.WaitForSingleObject(self.handle, 1000)
            if result == 258:
                continue
            if result != 0:
                raise ctypes.WinError(ctypes.get_last_error())
            code = wintypes.DWORD()
            if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if code.value:
                raise RuntimeError(f'Native predecessor exited with code {code.value}; no GPU work started')
            return code.value

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
