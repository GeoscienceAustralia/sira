"""
Params lists and functions for validation testing of models and config files
Validates model and config files based on rules
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths

SIRA_ROOT_DIR = Path(__file__).resolve().parent
path_to_test_models = Path(SIRA_ROOT_DIR, "tests", "models")

# -----------------------------------------------------------------------------
# Worksheet names / primary JSON keys

# XL_WORKSHEET_NAMES
required_model_worksheets = [
    'system_meta',
    'component_list',
    'component_connections',
    'supply_setup',
    'output_setup',
    'comp_type_dmg_algo',
    'damage_state_def'
]

# XL_COMPONENT_LIST_HEADERS
required_col_names_clist = [
    'component_id',
    'component_type',
    'component_class',
    'cost_fraction',
    'node_type',
    'node_cluster',
    'operating_capacity',
    'pos_x',
    'pos_y'
]

# MODEL_COMPONENT_HEADERS
required_component_headers = [
    "component_type",
    "component_class",
    "cost_fraction",
    "node_type",
    "node_cluster",
    "operating_capacity",
    "pos_x",
    "pos_y",
    "damages_states_constructor"
]

# MODEL_CONNECTION_HEADERS
required_col_names_conn = [
    'origin',
    'destination',
    'link_capacity',
    'weight'
]

# ALGORITHM_DEFINITION_PARAMS
required_col_names = [
    'is_piecewise',
    'damage_function',
    'damage_ratio',
    'functionality',
    'recovery_function',
    'recovery_param1',
    'recovery_param2'
]

# MODEL_SECTIONS
required_headers = [
    "component_list",
    "node_conn_df",
    "sysinp_setup",
    "sysout_setup"
]

# -----------------------------------------------------------------------------


def config_file_valid(config_file):
    """
    Config File Validation Rules:
    """
    return True


def model_file_valid(model_file):
    """
    Model Validation Rules:
    1. nodes in component list should appear in connections
    2. nodes should not be orphans
    3. component types should match between damage algorithms and component list
    """
    return True
