import faulthandler

import customtkinter as ctk
import gil_load
import torch
from scalene import scalene_profiler

from modules.util.ui import components


class ProfilingWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.title("Profiling")
        self.geometry("512x512")
        self.resizable(True, True)
        self.wait_visibility()
        self.focus_set()

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)
        self.grid_rowconfigure(4, weight=1)
        self.grid_columnconfigure(0, weight=1)

        components.button(self, 0, 0, "Dump stack", self._dump_stack)
        self._profile_button = components.button(
            self, 1, 0, "Start Profiling", self._start_profiler,
            tooltip="Turns on/off Scalene profiling. Only works when OneTrainer is launched with Scalene!")

        self._memory_button = components.button(
            self, 2, 0, "Start Memory Profiling", self._start_memory_profiler,
            tooltip="Turns on/off memory profiling.")

        self._gil_button = components.button(
            self, 3, 0, "Start GIL Profiling", self._start_gil_profiler,
            tooltip="Turns on/off GIL profiling.")

        # Bottom bar
        self._bottom_bar = ctk.CTkFrame(master=self, corner_radius=0)
        self._bottom_bar.grid(row=4, column=0, sticky="sew")
        self._message_label = components.label(self._bottom_bar, 0, 0, "Inactive")

        gil_load.init()

        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self.withdraw()

    def _dump_stack(self):
        with open('stacks.txt', 'w') as f:
            faulthandler.dump_traceback(f)
        self._message_label.configure(text='Stack dumped to stacks.txt')

    def _end_gil_profiler(self):
        gil_load.stop()
        stats = gil_load.get()
        with open('gil.txt', 'w') as f:
            f.write(gil_load.format(stats))
        self._message_label.configure(text='Stopped GIL profiling')
        self._gil_button.configure(text='Start GIL Profiling')
        self._gil_button.configure(command=self._start_gil_profiler)

    def _end_memory_profiler(self):
        try:
            torch.cuda.memory._dump_snapshot('memory_profile.pickle')
            torch.cuda.memory._record_memory_history(enabled=None)
            self._message_label.configure(
                text='Memory profile dumped to memory_profile.pickle')
        except Exception as e:
            self._message_label.configure(text='Failed to dump memory profile.')
        self._memory_button.configure(text='Start Memory Profiling')
        self._memory_button.configure(command=self._start_memory_profiler)

    def _end_profiler(self):
        scalene_profiler.stop()

        self._message_label.configure(text='Inactive')
        self._profile_button.configure(text='Start Profiling')
        self._profile_button.configure(command=self._start_profiler)

    def _start_gil_profiler(self):
        gil_load.test()
        gil_load.start()
        self._message_label.configure(text='GIL profiling active...')
        self._gil_button.configure(text='End GIL Profiling')
        self._gil_button.configure(command=self._stop_gil_profiler)

    def _start_memory_profiler(self):
        torch.cuda.memory._record_memory_history(max_entries=100000)
        self._message_label.configure(text='Memory profiling active...')
        self._memory_button.configure(text='End Memory Profiling')
        self._memory_button.configure(command=self._end_memory_profiler)

    def _start_profiler(self):
        scalene_profiler.start()

        self._message_label.configure(text='Profiling active...')
        self._profile_button.configure(text='End Profiling')
        self._profile_button.configure(command=self._end_profiler)
