"""
TorchLib module
"""


# pylint: disable=C0415
class TorchLib:
    """
    Imports torch dependencies with fallbacks when the library is not installed.
    """

    def dataset(self):
        """
        Import torch.utils.data.Dataset.

        Returns:
            Dataset
        """

        try:
            from torch.utils.data import Dataset

        except ImportError:

            class Dataset:
                """
                Stub for Dataset
                """

        return Dataset

    def module(self):
        """
        Imports torch.nn.Module

        Returns:
            Module
        """

        try:
            import torch.nn

            # pylint: disable=C0103
            Module = torch.nn.Module

        except ImportError:

            class Module:
                """
                Stub for Module
                """

        return Module

    def pretrained(self):
        """
        Imports transformers.modeling_utils.PreTrainedModel.

        Returns:
            PreTrainedModel
        """

        try:
            from transformers.modeling_utils import PreTrainedModel

        except ImportError:

            class PreTrainedModel:
                """
                Stub for PreTrainedModel
                """

        return PreTrainedModel

    def torch(self):
        """
        Imports torch.

        Returns:
            torch
        """

        try:
            import torch

        except ImportError:

            class Torch:
                """
                Stub for torch
                """

                def __getattr__(self, name):
                    raise ImportError("Torch is not installed, install torch to use this module")

            torch = Torch()

        return torch
