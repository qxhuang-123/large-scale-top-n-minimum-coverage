"""Read-only Windows process/system memory monitoring, without extra packages."""
import ctypes as C
from ctypes import wintypes as W


class Counters(C.Structure):
    _fields_ = [('cb', W.DWORD), ('PageFaultCount', W.DWORD)] + [
        (k, C.c_size_t) for k in ('PeakWorkingSetSize', 'WorkingSetSize',
        'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


class MemStatus(C.Structure):
    _fields_ = [('length', W.DWORD), ('load', W.DWORD)] + [
        (k, C.c_ulonglong) for k in ('total', 'available', 'pagefile', 'availpage',
        'virtual', 'availvirtual', 'extended')]


kernel = C.WinDLL('kernel32', use_last_error=True)
psapi = C.WinDLL('psapi', use_last_error=True)
kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
kernel.OpenProcess.restype = W.HANDLE
kernel.CloseHandle.argtypes = [W.HANDLE]
kernel.TerminateProcess.argtypes = [W.HANDLE, W.UINT]
kernel.TerminateProcess.restype = W.BOOL
kernel.WaitForSingleObject.argtypes = [W.HANDLE, W.DWORD]
kernel.WaitForSingleObject.restype = W.DWORD
psapi.GetProcessMemoryInfo.argtypes = [W.HANDLE, C.POINTER(Counters), W.DWORD]
psapi.GetProcessMemoryInfo.restype = W.BOOL


def available_memory():
    m = MemStatus(); m.length = C.sizeof(m)
    kernel.GlobalMemoryStatusEx(C.byref(m))
    return int(m.available), int(m.total)


class ProcessMemory:
    def __init__(self, pid):
        self.pid = pid
        self.handle = kernel.OpenProcess(0x0400 | 0x0010 | 0x0001 | 0x00100000, False, pid)
        if not self.handle:
            raise OSError(C.get_last_error(), 'Cannot monitor worker memory')

    def read(self):
        c = Counters(); c.cb = C.sizeof(c)
        if not psapi.GetProcessMemoryInfo(self.handle, C.byref(c), c.cb):
            return None
        return dict(working_set=int(c.WorkingSetSize), peak_working_set_os=int(c.PeakWorkingSetSize),
                    private_bytes=int(c.PrivateUsage), peak_commit_os=int(c.PeakPagefileUsage))

    def close(self):
        if self.handle:
            kernel.CloseHandle(self.handle); self.handle = None

    def terminate(self):
        if self.handle:
            kernel.TerminateProcess(self.handle, 1)
            kernel.WaitForSingleObject(self.handle, 5000)
