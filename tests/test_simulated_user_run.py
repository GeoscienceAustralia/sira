import unittest
import subprocess
from pathlib import Path

# Add the source dir to system path
import sys
src_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(src_dir))


class TestModuleRunProcess(unittest.TestCase):
    """
    Sets up expected paths, and runs tests to simulate how an user would 
    typically run the application from the terminal.
    """

    def setUp(self):
        self.root_dir = Path(__file__).resolve().parent.parent
        self.test_dir = Path(self.root_dir, 'tests')
        self.code_dir = Path(self.root_dir, 'sira')
        self.mdls_dir = Path(self.test_dir, 'models')

    def test_term_run_psmodel(self):
        """
        This module tests:
        running the application from the terminal,
        for a powerstation model.
        """
        model_name = 'powerstation_coal_A'
        inputdir = Path(self.mdls_dir, model_name)
        process = subprocess.run(
            ['python', str(self.code_dir), '-d', str(inputdir), '-s'], 
            stdout=subprocess.PIPE, 
            universal_newlines=True)
        exitstatus = process.returncode
        print(process.stdout)
        # An exit status of 0 typically indicates process ran successfully:
        self.assertEqual(exitstatus, 0)

    def test_full_term_run_wtpmodel(self):
        """
        This module tests:
        - running the application from the terminal, for
        - damage simulation to a water treatment plant model, followed by
        - fragility curve fitting, and
        - loss & recovery analysis.
        """
        model_name = 'potable_water_treatment_plant_A'
        inputdir = Path(self.mdls_dir, model_name)
        process = subprocess.run(
            ['python', str(self.code_dir), '-d', str(inputdir), '-sfl'], 
            stdout=subprocess.PIPE, 
            universal_newlines=True)
        exitstatus = process.returncode
        print(process.stdout)
        # An exit status of 0 typically indicates process ran successfully:
        self.assertEqual(exitstatus, 0)


if __name__ == '__main__':
    unittest.main()
