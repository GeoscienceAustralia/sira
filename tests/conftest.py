import sys
import pytest
from pathlib import Path


@pytest.fixture(scope="module", autouse=True)
def dir_setup():
    """A pytest fixture to set up test directory structure for SIRA."""
    root_dir = Path(__file__).resolve().parent.parent
    code_dir = Path(root_dir, 'sira')
    mdls_dir = Path(root_dir, 'tests', 'models')
    # Add the source dir to system path
    sys.path.insert(0, str(code_dir))
    return code_dir, mdls_dir
