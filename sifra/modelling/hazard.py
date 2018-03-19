import numpy as np


class Hazard(object):
    """
    A range of hazard levels that are created from the scenario values.

    The idea is to abstract the number and type of hazards to allow greater
    flexibility in the type and number of hazards to be modelled.
    """
    def __init__(self, configuration):
        """Create the list of hazard levels from the scenario values"""
        # str name of hazard
        self.hazard_type = configuration.HAZARD_TYPE

        self.num_hazard_pts = \
            int(round((configuration.PGA_MAX - configuration.PGA_MIN) /
                      float(configuration.PGA_STEP) + 1))

        # using the limits and step generate a list of hazard intensity values
        self.hazard_intensity_vals = np.linspace(configuration.PGA_MIN,
                                                 configuration.PGA_MAX,
                                                 num=self.num_hazard_pts)

        self.hazard_range = [HazardLevel(configuration, hazard_intensity)
                             for hazard_intensity in self.hazard_intensity_vals]

        self.intensity_measure_param = configuration.INTENSITY_MEASURE_PARAM
        self.intensity_measure_unit = configuration.INTENSITY_MEASURE_UNIT


class HazardLevel(object):
    """A particular level of hazard in the range of hazard levels."""
    def __init__(self, configuration, hazard_intensity):
        """Only the intensity is currently used."""
        self.hazard_type = configuration.HAZARD_TYPE

        if configuration.HAZARD_RASTER:
            self.level_factor_raster = configuration.HAZARD_RASTER
        else:
            self.level_factor_raster = None
        self.hazard_intensity = hazard_intensity

    def determine_intensity_at(self, location=None):
        if not location or (location.lon is np.NAN):
            return self.hazard_intensity
        else:
            # TODO Implement validation of parameters with the raster
            if not self.level_factor_raster:
                raise RuntimeError("A location was given, but a location raster has not been configured")
            # use the lat long to offset into the NetCDF
            return self.level_factor_raster.variables['pga_factor'][location.lat, location.lon]*self.hazard_intensity

