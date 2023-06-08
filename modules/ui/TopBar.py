import json
import os
from typing import Callable

import customtkinter as ctk

from modules.util import path_util
from modules.util.args.TrainArgs import TrainArgs
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.ui import components, dialogs
from modules.util.ui.UIState import UIState


class TopBar:
    def __init__(
            self,
            master,
            train_args:
            TrainArgs,
            ui_state: UIState,
            change_training_method_callback: Callable[[TrainingMethod], None]
    ):
        self.master = master
        self.train_args = train_args
        self.ui_state = ui_state
        self.change_training_method_callback = change_training_method_callback

        self.dir = "training_presets"

        self.config_ui_data = {
            "config_name": ""
        }
        self.config_ui_state = UIState(master, self.config_ui_data)

        self.configs = [("", "")]
        self.__load_available_config_names()

        self.current_config = []

        self.frame = ctk.CTkFrame(master=master, corner_radius=0)
        self.frame.grid(row=0, column=0, sticky="nsew")

        # title
        components.app_title(self.frame, 0, 0)

        # dropdown
        self.configs_dropdown = None
        self.__create_configs_dropdown()

        # remove button
        components.icon_button(self.frame, 0, 2, "-", self.__remove_config)

        # save button
        components.button(self.frame, 0, 3, "save current", self.__save_config)

        # padding
        self.frame.grid_columnconfigure(4, weight=1)

        # training method
        components.options_kv(
            self.frame,
            row=0,
            column=5,
            values=[
                ("Fine Tune", TrainingMethod.FINE_TUNE),
                ("LoRA", TrainingMethod.LORA),
                ("Embedding", TrainingMethod.EMBEDDING),
                ("Fine Tune VAE", TrainingMethod.FINE_TUNE_VAE),
            ],
            ui_state=self.ui_state,
            var_name="training_method",
            command=self.change_training_method_callback
        )

    def __create_configs_dropdown(self):
        if self.configs_dropdown is not None:
            self.configs_dropdown.grid_forget()

        self.configs_dropdown = components.options_kv(
            self.frame, 0, 1, self.configs, self.config_ui_state, "config_name", self.__load_current_config
        )

    def __load_available_config_names(self):
        if os.path.isdir(self.dir):
            for path in os.listdir(self.dir):
                path = path_util.canonical_join(self.dir, path)
                if path.endswith(".json") and os.path.isfile(path):
                    name = os.path.basename(path)
                    name = os.path.splitext(name)[0]
                    self.configs.append((name, path))

    def __save_config(self):
        def create_file(name):
            name = path_util.safe_filename(name)
            path = path_util.canonical_join("training_presets", f"{name}.json")
            with open(path, "w") as f:
                json.dump(self.train_args.to_json(), f, indent=4)

            if name not in [x[0] for x in self.configs]:
                self.configs.append((name, path))

            self.__create_configs_dropdown()

        dialogs.StringInputDialog(self.master, "name", "Config Name", create_file)

    def __load_current_config(self, filename):
        try:
            with open(filename, "r") as f:
                self.train_args.from_json(json.load(f))
                self.ui_state.update(self.train_args)
        except Exception as e:
            print(e)

    def __remove_config(self):
        pass