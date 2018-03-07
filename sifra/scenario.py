import numpy as np


class Scenario:
    """
    Defines the scenario for hazard impact modelling
    It hold the required constants that are going to be used by the simulation
    """

    def __init__(self, configuration):

        self.input_dir_name = configuration.INPUT_DIR_NAME
        self.raw_output_dir = configuration.RAW_OUTPUT_DIR

        self.fit_pe_data = configuration.FIT_PE_DATA
        self.fit_restoration_data = configuration.FIT_RESTORATION_DATA

        self.output_path = configuration.OUTPUT_PATH

        #need to convert excel doc into json and update the ingest class
        self.algorithm_factory = None

        self.num_samples = configuration.NUM_SAMPLES
        self.save_vars_npy = configuration.SAVE_VARS_NPY
        self.haz_param_max = configuration.PGA_MAX
        self.haz_param_min = configuration.PGA_MIN
        self.haz_param_step = configuration.PGA_STEP

        self.hazard_type = configuration.HAZARD_TYPE
        self.intensity_measure_param = configuration.INTENSITY_MEASURE_PARAM
        self.intensity_measure_unit = configuration.INTENSITY_MEASURE_UNIT
        self.level_factor_raster = configuration.HAZARD_RASTER

        self.run_context = configuration.RUN_CONTEXT
        self.run_parallel_proc = configuration.MULTIPROCESS
        """Set up parameters for simulating hazard impact"""
        self.num_hazard_pts = \
            int(round((self.haz_param_max - self.haz_param_min) /
                      float(self.haz_param_step) + 1))

        self.hazard_intensity_vals = \
            np.linspace(self.haz_param_min, self.haz_param_max,
                        num=self.num_hazard_pts)
        self.hazard_intensity_str = \
            [('%0.3f' % np.float(x)) for x in self.hazard_intensity_vals]

        # Set up parameters for simulating recovery from hazard impact
        self.restoration_time_range, self.time_step = np.linspace(
            0, configuration.RESTORE_TIME_MAX, num=configuration.RESTORE_TIME_MAX + 1,
            endpoint=True, retstep=True)

        self.num_time_steps = len(self.restoration_time_range)

        self.restoration_chkpoints, self.restoration_pct_steps = \
            np.linspace(0.0, 1.0, num=configuration.RESTORE_PCT_CHECKPOINTS, retstep=True)
