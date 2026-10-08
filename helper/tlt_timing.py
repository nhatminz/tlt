"""Stream event timing: waits only for the interval being reported, never device sync."""
import torch


class StreamTimer:
    def __init__(self, device):
        self.device = device
        self.start = torch.cuda.Event(enable_timing=True)
        self.stop = torch.cuda.Event(enable_timing=True)

    def begin(self):
        self.start.record(torch.cuda.current_stream(self.device))

    def end(self):
        self.stop.record(torch.cuda.current_stream(self.device))

    def seconds(self):
        self.stop.synchronize()
        return self.start.elapsed_time(self.stop) / 1000.
