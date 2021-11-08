import subprocess
import pytest
from pathlib import Path

# ------------------------------------------------------------------------------

testdata = [
    ('_', 'powerstation_coal_A', '-s'),
    ('_', 'substation_tx_230kv', '-sl'),
    ('_', 'pumping_station_testbed', '-s'),
    ('_', 'potable_water_treatment_plant_A', '-sfl'),
    ('_', 'test_network__basic', '-s'),
    ('_', 'test_structure__parallel_piecewise', '-s')
]

files_with_incorrect_names = [
    ('_', 'test_invalid_modelname', '-s'),
    ('_', 'test_invalid_config', '-s'),
    ('_', 'test_missing_modelfile', '-s'),
    ('_', 'test_missing_configfile', '-s')
]

input_files_missing = [
    ('_', 'test_missing_inputdir', '-s')
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
    directly running models from the terminal.
    """
    code_dir, mdls_dir = dir_setup
    inputdir = Path(mdls_dir, model_name)
    cmd = ['python', str(code_dir), '-d', str(inputdir), run_arg]

    # process = subprocess.run(
    #     cmd, stdout=subprocess.PIPE, universal_newlines=True, check=False)
    process = subprocess.run(
        cmd, universal_newlines=True, check=False)
    exitstatus = process.returncode

    # An exit status of 0 typically indicates process ran successfully:
    assert exitstatus == 0, f"Run failed for {model_name} with args {run_arg}"


@pytest.mark.bad_or_missing_inputfile
@pytest.mark.parametrize(
    "dir_setup, model_name, run_arg",
    files_with_incorrect_names,
    indirect=["dir_setup"])
def test_catch_improper_inpufilename(dir_setup, model_name, run_arg):
    code_dir, mdls_dir = dir_setup
    inputdir = Path(mdls_dir, model_name)
    cmd = ['python', str(code_dir), '-d', str(inputdir), run_arg]

    proc = subprocess.run(
        cmd, capture_output=True, universal_newlines=True, check=False)
    assert proc.returncode == 1
    assert "invalid or missing input file" in str(proc.stderr)


@pytest.mark.missinginputdir
@pytest.mark.parametrize(
    "dir_setup, model_name, run_arg",
    input_files_missing,
    indirect=["dir_setup"])
def test_missing_inputdir(dir_setup, model_name, run_arg):
    code_dir, mdls_dir = dir_setup
    inputdir = Path(mdls_dir, model_name)
    err_msg = "Invalid path"
    cmd = ['python', str(code_dir), '-d', str(inputdir), run_arg]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
        shell=False
    )
    msg_out = str(proc.stdout).lower() + str(proc.stderr).lower()

    print("*" * 80)
    print(f"Input dir:\n {str(inputdir)}")
    print("*" * 80)
    print(f"combined msg:\n {msg_out}")
    print("*" * 80)

    assert proc.returncode == 1
    assert err_msg.lower() in msg_out


if __name__ == '__main__':
    pytest.main([__file__])
