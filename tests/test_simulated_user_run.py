import subprocess
import pytest
from pathlib import Path

# ------------------------------------------------------------------------------

testdata = [
    ('', 'powerstation_coal_A', '-s'),
    ('', 'substation_tx_230kv', '-sl'),
    ('', 'potable_water_treatment_plant_A', '-sfl'),
    ('', 'test_network__basic', '-s')
]

# ------------------------------------------------------------------------------

@pytest.mark.modelrun
@pytest.mark.parametrize(
    "dir_setup, model_name, run_arg",
    testdata,
    indirect=["dir_setup"])
def test_run_model(dir_setup, model_name, run_arg):
    """
    This module tests:
    running the application from the terminal,
    for a powerstation model.
    """

    code_dir, mdls_dir = dir_setup

    inputdir = Path(mdls_dir, model_name)
    cmd = ['python', str(code_dir), '-d', str(inputdir), run_arg]

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True)
    exitstatus = process.returncode

    # An exit status of 0 typically indicates process ran successfully:
    assert exitstatus == 0, f"Run failed for {model_name} with args {run_arg}"

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__])
