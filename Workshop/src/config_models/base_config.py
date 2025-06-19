# This file defines the BaseConfig class, which serves as a foundational Pydantic model
# for all other configuration classes in the application. It provides common functionality
# for loading configurations from JSON files and saving them back to JSON.
#
# Class: BaseConfig
# Methods:
#  - from_json(cls: Type[T], path: str | Path) -> T:
#    Loads a configuration object from a JSON file.
#  - to_json(self, path: str | Path) -> None:
#    Saves the configuration object to a JSON file.
#  - _get_config_file_path(config_name: str, config_type_directory: str) -> str: (static)
#    Constructs the standardized path to a configuration file.
#  - load_from_name(cls: Type[T], config_name: str, config_type_directory: str) -> T:
#    Loads a configuration object by its name and type directory, resolving the file path.

import json
import os  # Added for os.path.join and os.path.exists
from pathlib import Path
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="BaseConfig")


class BaseConfig(BaseModel):
    @classmethod
    def from_json(cls: Type[T], path: str | Path) -> T:
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=4)

    @staticmethod
    def _get_config_file_path(config_name: str, config_type_directory: str) -> str:
        """Constructs the path to a configuration file."""
        if not config_name.endswith(".json"):
            config_name += ".json"
        # Assumes this is run from the project root or that "configs" is accessible.
        return os.path.join("configs", config_type_directory, config_name)

    @classmethod
    def load_from_name(cls: Type[T], config_name: str, config_type_directory: str) -> T:
        """Loads a configuration object by its name and configuration type directory."""
        file_path = cls._get_config_file_path(config_name, config_type_directory)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Configuration file not found for {cls.__name__} '{config_name}' at {file_path}. "
                f"Ensure the file exists, the name is correct, and you are running from the project root."
            )
        return cls.from_json(file_path)
