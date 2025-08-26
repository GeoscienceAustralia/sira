import numpy as np


class Scenario:
    """
    Defines the scenario for hazard impact modelling.
    It holds the required constants that are going to be used in the simulation.
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

        # Recovery analysis configuration
        self.recovery_method = getattr(configuration, 'RECOVERY_METHOD', 'max')
        self.num_repair_streams = getattr(configuration, 'NUM_REPAIR_STREAMS', 100)

        # Optional parameters - None means use all available/auto-calculate
        self.recovery_max_workers = getattr(configuration, 'RECOVERY_MAX_WORKERS', None)
        self.recovery_batch_size = getattr(configuration, 'RECOVERY_BATCH_SIZE', None)

        self.restoration_checkpoints, self.restoration_pct_steps \
            = np.linspace(
                0.0, 1.0, num=configuration.RESTORE_PCT_CHECKPOINTS, retstep=True
            )

        # Parallel computing attributes
        # These will be set by the main module when parallel processing is enabled
        # NOTE: These are kept as None by default and excluded from pickle operations
        self._parallel_config = None
        self._parallel_backend_data = None

        # List of attributes to exclude from pickling (non-serialisable objects)
        self._pickle_exclude = {
            '_parallel_config',
            '_parallel_backend_data',
            'recovery_analysis_time'
        }

        # Recovery analysis timing - set during recovery analysis
        self.recovery_analysis_time = None

    def __getstate__(self):
        """
        Custom pickling method that excludes non-serialisable attributes.
        This ensures the object can be safely pickled for multiprocessing.
        """
        # Get the object's state (all attributes)
        state = self.__dict__.copy()

        # Remove non-serialisable attributes
        for attr in self._pickle_exclude:
            if attr in state:
                state.pop(attr)

        return state

    def __setstate__(self, state):
        """
        Custom unpickling method that restores the object and sets
        excluded attributes to None.
        """
        # Restore the object's state
        self.__dict__.update(state)

        # Ensure excluded attributes exist and are set to None
        for attr in self._pickle_exclude:
            if not hasattr(self, attr):
                setattr(self, attr, None)

    @property
    def parallel_config(self):
        """Access parallel config with safe fallback."""
        return getattr(self, '_parallel_config', None)

    @parallel_config.setter
    def parallel_config(self, value):
        """Set parallel config."""
        self._parallel_config = value

    @property
    def parallel_backend_data(self):
        """Access parallel backend data with safe fallback."""
        return getattr(self, '_parallel_backend_data', None)

    @parallel_backend_data.setter
    def parallel_backend_data(self, value):
        """Set parallel backend data."""
        self._parallel_backend_data = value

    def set_parallel_config(self, parallel_config, backend_data=None):
        """
        Set parallel computing configuration.

        Parameters
        ----------
        parallel_config : ParallelConfig
            Parallel configuration object
        backend_data : dict, optional
            Backend-specific data
        """
        self._parallel_config = parallel_config
        self._parallel_backend_data = backend_data

    def get_recovery_config(self):
        """
        Get recovery analysis configuration.

        Returns
        -------
        dict
            Recovery configuration parameters
        """
        return {
            'recovery_method': self.recovery_method,
            'num_repair_streams': self.num_repair_streams,
            'recovery_max_workers': self.recovery_max_workers,
            'recovery_batch_size': self.recovery_batch_size
        }

    def set_recovery_analysis_time(self, analysis_time):
        """
        Set the recovery analysis execution time.

        Parameters
        ----------
        analysis_time : float
            Time taken for recovery analysis in seconds
        """
        self.recovery_analysis_time = analysis_time

    def is_parallel_enabled(self):
        """
        Check if parallel processing is enabled.

        Returns
        -------
        bool
            True if parallel processing is configured
        """
        return self.parallel_config is not None

    def get_parallel_backend(self):
        """
        Get the configured parallel backend.

        Returns
        -------
        str or None
            Backend name ('mpi', 'dask', 'multiprocessing') or None if not configured
        """
        if self.parallel_config:
            return self.parallel_config.config.get('backend', None)
        return None

    def create_worker_copy(self):
        """
        Create a copy of this scenario suitable for worker processes.
        This ensures that parallel-specific attributes are properly handled.

        Returns
        -------
        Scenario
            A clean copy suitable for multiprocessing
        """
        import copy

        # Create a shallow copy
        worker_scenario = copy.copy(self)

        # Reset parallel-specific attributes to None for workers
        worker_scenario._parallel_config = None
        worker_scenario._parallel_backend_data = None
        worker_scenario.recovery_analysis_time = None

        return worker_scenario
