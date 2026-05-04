"""
Torch module tests
"""

import sys
import unittest

# pylint: disable=C0415,W0611,W0621
import txtai


class TestTorch(unittest.TestCase):
    """
    Simulates torch not being installed. Even though torch is a required dependency, txtai can operate without it gracefully.
    """

    @classmethod
    def setUpClass(cls):
        """
        Simulate torch not being installed
        """

        modules = [
            "transformers.modeling_utils",
            "transformers.modeling_outputs",
            "torch",
            "torch.nn",
            "torch.onnx",
            "torch.utils.data",
        ]

        # Get handle to all currently loaded txtai modules
        modules = modules + [key for key in sys.modules if key.startswith("txtai")]
        cls.modules = {module: None for module in modules}

        # Replace loaded modules with stubs. Save modules for later reloading
        for module in cls.modules:
            if module in sys.modules:
                cls.modules[module] = sys.modules[module]

            # Remove txtai modules. Set optional dependencies to None to prevent reloading.
            if "txtai" in module:
                if module in sys.modules:
                    del sys.modules[module]
            else:
                sys.modules[module] = None

    @classmethod
    def tearDownClass(cls):
        """
        Resets modules environment back to initial state.
        """

        # Reset replaced modules in setup
        for key, value in cls.modules.items():
            if value:
                sys.modules[key] = value
            else:
                del sys.modules[key]

    def testTorch(self):
        """
        Test torch not installed
        """

        from txtai.util import TorchLib

        torchlib = TorchLib()

        # Test torch stubs
        self.assertTrue(torchlib.dataset().__module__.endswith("torchlib"))
        self.assertTrue(torchlib.module().__module__.endswith("torchlib"))
        self.assertTrue(torchlib.pretrained().__module__.endswith("torchlib"))
        self.assertTrue(torchlib.torch())

        # pylint: disable=W0106
        with self.assertRaises(ImportError):
            torchlib.torch().device
