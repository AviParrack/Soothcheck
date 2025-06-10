from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import json
from typing import Optional, Generic, TypeVar, TYPE_CHECKING, get_args, get_origin, Any
import importlib

from .config import (
        ProbeConfig,
        LinearProbeConfig,
        LogisticProbeConfig,
        MultiClassLogisticProbeConfig,
        KMeansProbeConfig,
        PCAProbeConfig,
        MeanDiffProbeConfig,
        SklearnLogisticProbeConfig,
        LogisticProbeConfigBase,
    )

T = TypeVar("T", bound="ProbeConfig")

def _get_config_attr(config, attr_name, default=None):
    return getattr(config, attr_name, default)

def _set_config_attr(config, attr_name, value):
    if hasattr(config, attr_name):
        setattr(config, attr_name, value)

class BaseProbe(ABC, nn.Module, Generic[T]):
    """Abstract base class for probes. Probes store directions in the original activation space."""

    config: T  # Type hint for config instance

    def __init__(self, config: T):
        super().__init__()
        self.config = config
        dtype_str = _get_config_attr(config, "dtype", "float32")
        
        if dtype_str == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype_str == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        
        self.name = (
            _get_config_attr(config, "name", "unnamed_probe") or "unnamed_probe"
        )
    
    @abstractmethod
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction in the original activation space.

        Args:
            normalized: Whether to normalize the direction vector to unit length.
                      The probe's internal configuration (`normalize_weights`)
                      also influences this. Normalization occurs only if
                      `normalized` is True AND `config.normalize_weights` is True.

        Returns:
            The processed (optionally normalized) direction vector
            representing the probe in the original activation space.
        """
        pass

    @abstractmethod
    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the raw internal representation (weights/vector) before normalization."""
        pass

    @abstractmethod
    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the raw internal representation (weights/vector) from a (potentially adjusted) vector."""
        pass

    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot product between activations and the probe direction."""
        direction = self.get_direction(normalized=True)
        acts = acts.to(dtype=self.dtype)
        direction = direction.to(dtype=self.dtype)
        return torch.einsum("...d,d->...", acts, direction)

    def save(self, path: str) -> None:
        """Save probe state and config in a single .pt file."""
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        config_dict = self.config.__dict__.copy()
        additional_info = config_dict.get("additional_info", {})

        additional_info.pop("is_standardized", None)
        additional_info.pop("feature_mean", None)
        additional_info.pop("feature_std", None)

        has_bias_param = False
        if hasattr(self, "linear") and isinstance(self.linear, nn.Module):
            if hasattr(self.linear, "bias") and self.linear.bias is not None:
                has_bias_param = True
        elif hasattr(self, "intercept_"):
            has_bias_param = self.intercept_ is not None

        additional_info["has_bias"] = has_bias_param

        if hasattr(self.config, "normalize_weights"):
            additional_info["normalize_weights"] = _get_config_attr(
                self.config, "normalize_weights"
            )
        if hasattr(self.config, "bias"):
            additional_info["bias_config"] = _get_config_attr(self.config, "bias")

        config_dict["additional_info"] = additional_info

        config_to_save = type(self.config)(**config_dict)

        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": config_to_save,
                "probe_type": self.__class__.__name__,
            },
            path,
        )

    def save_json(self, path: str) -> None:
        """Save probe's internal direction and metadata as JSON."""
        if not path.endswith(".json"):
            path += ".json"
    
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
        try:
            vector = self._get_raw_direction_representation()
            if vector is None:
                raise ValueError("Raw direction representation is None.")
            
            if vector.dtype == torch.bfloat16:
                vector = vector.to(torch.float32)
            
            vector_np = vector.detach().clone().cpu().numpy()
        except Exception as e:
            print(f"Error getting raw direction for {self.name}: {e}. Cannot save JSON.")
            return

        metadata = self._prepare_metadata(vector_np)

        save_data = {
            "vector": vector_np.tolist(),
            "metadata": metadata,
        }

        try:
            with open(path, "w") as f:
                json.dump(save_data, f, indent=2)
        except TypeError as e:
            print(f"Error serializing probe data to JSON for {self.name}: {e}")
            try:
                import numpy as np

                def default_serializer(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu().numpy().tolist()
                    return str(obj)

                with open(path, "w") as f:
                    json.dump(save_data, f, indent=2, default=default_serializer)
                print("Successfully saved JSON with fallback serializer.")
            except Exception as final_e:
                print(
                    f"Fallback JSON serialization also failed for {self.name}: {final_e}"
                )

    def _prepare_metadata(self, vector_np: Any) -> dict:
        """Helper to prepare metadata dictionary for saving."""
        metadata = {
            "model_name": _get_config_attr(self.config, "model_name", "unknown_model"),
            "hook_point": _get_config_attr(self.config, "hook_point", "unknown_hook"),
            "hook_layer": _get_config_attr(self.config, "hook_layer", 0),
            "hook_head_index": _get_config_attr(self.config, "hook_head_index"),
            "name": self.name,
            "vector_dimension": (
                vector_np.shape[-1] if hasattr(vector_np, "shape") else None
            ),
            "probe_type": self.__class__.__name__,
            "dataset_path": _get_config_attr(self.config, "dataset_path"),
            "prepend_bos": _get_config_attr(self.config, "prepend_bos", True),
            "context_size": _get_config_attr(self.config, "context_size", 128),
            "dtype": _get_config_attr(self.config, "dtype", "float32"),
            "device": _get_config_attr(self.config, "device"),
        }
    
        bias_value = None
        has_bias_param = False
        if hasattr(self, "linear") and isinstance(self.linear, nn.Module):
            if hasattr(self.linear, "bias") and self.linear.bias is not None:
                bias_param = self.linear.bias
                if isinstance(bias_param, torch.Tensor):
                    if bias_param.dtype == torch.bfloat16:
                        bias_param = bias_param.to(torch.float32)
                    bias_value = bias_param.data.detach().clone().cpu().numpy().tolist()
                    has_bias_param = True
        elif hasattr(self, "intercept_") and self.intercept_ is not None:
            intercept_param = self.intercept_
            if isinstance(intercept_param, torch.Tensor):
                if intercept_param.dtype == torch.bfloat16:
                    intercept_param = intercept_param.to(torch.float32)
                bias_value = intercept_param.data.detach().clone().cpu().numpy().tolist()
                has_bias_param = True
    
        metadata["has_bias"] = has_bias_param
        if has_bias_param:
            metadata["bias"] = bias_value

        config_flags_to_save = [
            "normalize_weights",
            "bias",
            "loss_type",
            "n_clusters",
            "n_init",
            "random_state",
            "n_components",
            "standardize",
            "max_iter",
            "solver",
        ]
        for flag in config_flags_to_save:
            if hasattr(self.config, flag):
                metadata_key = "bias_config" if flag == "bias" else flag
                metadata[metadata_key] = _get_config_attr(self.config, flag)

        additional_info = _get_config_attr(self.config, "additional_info", {})
        if isinstance(additional_info, dict):
            metadata.update(additional_info)

        return metadata

    @classmethod
    def load_json(cls, path: str, device: Optional[str] = None) -> "BaseProbe":
        """Load probe from JSON file.

        Args:
            path: Path to the JSON file
            device: Optional device override. If None, uses device from metadata or default.

        Returns:
            Loaded probe instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved probe JSON file found at {path}")

        with open(path, "r") as f:
            data = json.load(f)

        vector_list = data.get("vector")
        metadata = data.get("metadata", {})
        if vector_list is None:
            raise ValueError(f"JSON file {path} missing 'vector' field.")

        target_device_str = (
            device
            or metadata.get("device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        target_device = torch.device(target_device_str)

        probe_type_name = metadata.get("probe_type", cls.__name__)
        probe_cls = cls._get_probe_class_by_name(probe_type_name)
        config_cls = cls._get_config_class_for_probe(probe_cls, probe_type_name)

        dim = metadata.get("vector_dimension")
        if dim is None:
            try:
                temp_vector = torch.tensor(vector_list)
                dim = temp_vector.shape[-1]
                print(
                    f"Warning: vector_dimension not found in metadata, inferred as {dim} from vector shape."
                )
            except Exception as e:
                raise ValueError(
                    f"Could not determine vector dimension from metadata or vector in {path}: {e}"
                )

        try:
            config = config_cls(input_size=dim)
        except TypeError as e:
            raise TypeError(
                f"Error instantiating config class {config_cls.__name__} with input_size={dim}. Is it the correct config class for {probe_type_name}? Error: {e}"
            )

        cls._update_config_from_metadata(config, metadata, target_device_str)

        probe = probe_cls(config)
        probe.to(target_device)

        try:
            vector_tensor = torch.tensor(vector_list, dtype=probe.dtype).to(
                target_device
            )
            probe._set_raw_direction_representation(vector_tensor)
        except Exception as e:
            raise ValueError(f"Error processing 'vector' data from {path}: {e}")

        cls._restore_bias_intercept(probe, metadata, target_device)

        probe.eval()

        return probe

    @classmethod
    def _get_probe_class_by_name(cls, probe_type_name: str) -> type["BaseProbe"]:
        """Dynamically imports and returns the probe class."""
        try:
            module_name_part = probe_type_name.lower().replace("probe", "")
            if not module_name_part:
                raise ImportError("Could not determine module name part.")
            if "sklearn" in module_name_part:
                module_name_part = "sklearn_logistic"
            elif module_name_part == "linear":
                module_name_part = "linear"
            elif module_name_part == "logistic" or module_name_part == "multiclasslogistic":
                module_name_part = "logistic"
            elif module_name_part in ["kmeans", "pca", "meandifference", "directional"]:
                module_name_part = "directional"

            package_name = "probity.probes"
            module_full_name = f"{package_name}.{module_name_part}"
            probe_module = importlib.import_module(module_full_name)
            probe_cls = getattr(probe_module, probe_type_name)

            if not issubclass(probe_cls, BaseProbe):
                raise TypeError(
                    f"{probe_type_name} found in {module_full_name} is not a subclass of BaseProbe"
                )
            return probe_cls
        except (ImportError, AttributeError, TypeError, ValueError) as e:
            print(
                f"Warning: Could not dynamically load probe class {probe_type_name} from module {module_full_name if 'module_full_name' in locals() else 'unknown'}. Error: {e}. Falling back to {cls.__name__}."
            )
            if issubclass(cls, BaseProbe):
                return cls
            else:
                raise ImportError(
                    f"Fallback class {cls.__name__} is not a valid BaseProbe subclass."
                )

    @classmethod
    def _get_config_class_for_probe(
        cls, probe_cls: type["BaseProbe"], probe_type_name: str
    ) -> type["ProbeConfig"]:
        """Finds the corresponding config class for a given probe class."""
        config_cls = None
        config_cls_name = f"{probe_type_name}Config"
        try:
            config_module = importlib.import_module(".config", package="probity.probes")
            config_cls = getattr(config_module, config_cls_name)
            if issubclass(config_cls, ProbeConfig):
                return config_cls
        except (ImportError, AttributeError):
            pass

        try:
            for base in getattr(probe_cls, "__orig_bases__", []):
                if get_origin(base) is BaseProbe:
                    config_arg = get_args(base)[0]
                    if isinstance(config_arg, TypeVar):
                        pass
                    elif isinstance(config_arg, str):
                        try:
                            config_module = importlib.import_module(
                                ".config", package="probity.probes"
                            )
                            config_cls = eval(
                                config_arg, globals(), config_module.__dict__
                            )
                        except NameError:
                            pass
                    elif isinstance(config_arg, type) and issubclass(
                        config_arg, ProbeConfig
                    ):
                        config_cls = config_arg

                    if config_cls and issubclass(config_cls, ProbeConfig):
                        return config_cls
                    break
        except Exception as e:
            print(
                f"Warning: Exception while inferring config type from Generic hint for {probe_type_name}: {e}"
            )

        print(
            f"Warning: Could not determine specific config class for {probe_type_name}. Using base ProbeConfig."
        )
        try:
            config_module = importlib.import_module(".config", package="probity.probes")
            return getattr(config_module, "ProbeConfig")
        except (ImportError, AttributeError):
            raise ImportError(
                "Fatal: Could not load even the base ProbeConfig class from probity.probes.config"
            )

    @classmethod
    def _update_config_from_metadata(
        cls, config: "ProbeConfig", metadata: dict, target_device_str: str
    ) -> None:
        """Populates the config object with values from the metadata dict."""
        common_fields = [
            "model_name",
            "hook_point",
            "hook_layer",
            "hook_head_index",
            "name",
            "dataset_path",
            "prepend_bos",
            "context_size",
            "dtype",
        ]
        for key in common_fields:
            if key in metadata:
                _set_config_attr(config, key, metadata[key])

        _set_config_attr(config, "device", target_device_str)

        specific_metadata_keys = (
            set(metadata.keys())
            - set(common_fields)
            - {"device", "probe_type", "vector_dimension", "has_bias", "bias"}
        )

        for key in specific_metadata_keys:
            config_key = "bias" if key == "bias_config" else key
            _set_config_attr(config, config_key, metadata[key])

        if not hasattr(config, "additional_info") or config.additional_info is None:
            if isinstance(config, ProbeConfig):
                config.additional_info = {}

        if (
            "has_bias" in metadata
            and hasattr(config, "additional_info")
            and isinstance(config.additional_info, dict)
        ):
            config.additional_info["has_bias"] = metadata["has_bias"]

    @classmethod
    def _restore_bias_intercept(
        cls, probe: "BaseProbe", metadata: dict, target_device: torch.device
    ) -> None:
        """Restores bias/intercept from metadata if available."""
        if metadata.get("has_bias", False) and "bias" in metadata:
            bias_or_intercept_data = metadata["bias"]
            if bias_or_intercept_data is None:
                print(
                    f"Warning: 'has_bias' is true but 'bias' data is null in metadata for {probe.name}."
                )
                return

            try:
                tensor_data = torch.tensor(
                    bias_or_intercept_data, dtype=probe.dtype
                ).to(target_device)
            except Exception as e:
                print(
                    f"Warning: Could not convert bias/intercept metadata to tensor for {probe.name}: {e}"
                )
                return

            restored = False
            if (
                hasattr(probe, "linear")
                and isinstance(probe.linear, nn.Module)
                and hasattr(probe.linear, "bias")
            ):
                if probe.linear.bias is not None:
                    with torch.no_grad():
                        try:
                            if tensor_data.shape == probe.linear.bias.shape:
                                probe.linear.bias.copy_(tensor_data)
                                restored = True
                            else:
                                print(
                                    f"Warning: Bias shape mismatch during load. Metadata: {tensor_data.shape}, Probe: {probe.linear.bias.shape}. Attempting reshape."
                                )
                                probe.linear.bias.copy_(
                                    tensor_data.reshape(probe.linear.bias.shape)
                                )
                                restored = True
                        except Exception as e:
                            print(
                                f"Warning: Could not copy bias data to linear layer for {probe.name}: {e}"
                            )
                else:
                    print(
                        f"Warning: Bias metadata found for {probe.name}, but probe.linear.bias is None."
                    )

            if not restored and hasattr(probe, "intercept_"):
                with torch.no_grad():
                    try:
                        if probe.intercept_ is not None:
                            if tensor_data.shape == probe.intercept_.shape:
                                probe.intercept_.copy_(tensor_data)
                                restored = True
                            else:
                                print(
                                    f"Warning: Intercept shape mismatch. Metadata: {tensor_data.shape}, Probe: {probe.intercept_.shape}. Attempting reshape."
                                )
                                probe.intercept_.copy_(
                                    tensor_data.reshape(probe.intercept_.shape)
                                )
                                restored = True
                        else:
                            print(
                                f"Warning: Intercept buffer 'intercept_' was None for {probe.name}. Creating buffer from metadata."
                            )
                            probe.register_buffer("intercept_", tensor_data.clone())
                            restored = True
                    except Exception as e:
                        print(
                            f"Warning: Could not copy/register intercept data for {probe.name}: {e}"
                        )

            if not restored:
                print(
                    f"Warning: Bias/intercept metadata was present for {probe.name} but could not be restored to either linear.bias or intercept_."
                )