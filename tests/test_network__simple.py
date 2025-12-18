import subprocess
import sys
import unittest
from pathlib import Path

# Add the source dir to system path
root_dir = Path(__file__).resolve().parent.parent
code_dir = Path(root_dir, "sira")
sys.path.insert(0, str(code_dir))


class TestNetworkModelling(unittest.TestCase):
    """
    Sets up expected paths, and runs tests to simulate how an user would
    typically run the application from the terminal.
    """

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.code_dir = Path(self.root_dir, "sira")
        self.test_dir = Path(self.root_dir, "tests")
        self.mdls_dir = Path(self.test_dir, "models")

    def test_network_model_run(self):
        """
        This module tests:
        loading config file for network models
        """
        model_name = "test_network__basic"
        target_mdl_dir = Path(self.mdls_dir, model_name)

        process = subprocess.run(
            [sys.executable, "-m", "sira", "-d", str(target_mdl_dir), "-s"],
            stdout=subprocess.PIPE,
            universal_newlines=True,
            check=True,
        )
        exitstatus = process.returncode
        # An exit status of 0 typically indicates process ran successfully:
        self.assertEqual(exitstatus, 0)


if __name__ == "__main__":
    unittest.main()
