import numpy as np


class Scenario:
    """
    Defines the scenario for hazard impact modelling
    It holds the required constants that are going to be used in the simulation
    """

    def __init__(self, configuration):

        self.infrastructure_level = configuration.INFRASTRUCTURE_LEVEL
        self.raw_output_dir = configuration.RAW_OUTPUT_DIR
        self.output_path = configuration.OUTPUT_DIR
        self.hazard_input_method = configuration.HAZARD_INPUT_METHOD

        # need to convert excel doc into json and update the ingest class
        self.algorithm_factory = None

        self.fit_restoration_data = configuration.SWITCH_FIT_RESTORATION_DATA
        if configuration.SWITCH_SAVE_VARS_NPY in [1, "1", True, "True"]:
            self.save_vars_npy = True
        else:
            self.save_vars_npy = False
        self.run_context = configuration.RUN_CONTEXT
        self.run_parallel_proc = configuration.MULTIPROCESS

        self.num_samples = configuration.NUM_SAMPLES

        # Set up parameters for simulating recovery from hazard impact
        self.time_unit = configuration.TIME_UNIT
        self.restoration_streams = configuration.RESTORATION_STREAMS
        self.restoration_time_range, self.time_step \
            = np.linspace(
                0, configuration.RESTORE_TIME_MAX,
                num=configuration.RESTORE_TIME_MAX + 1,
                endpoint=True,
                retstep=True
            )
        self.num_time_steps = len(self.restoration_time_range)

        self.restoration_checkpoints, self.restoration_pct_steps \
            = np.linspace(
                0.0, 1.0, num=configuration.RESTORE_PCT_CHECKPOINTS, retstep=True
            )
