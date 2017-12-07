import numpy as np


class HazardLevels(object):
    """
    A range of hazard levels that are created from the scenario values.

    The idea is to abstract the number and type of hazards to allow greater
    flexibility in the type and number of hazards to be modelled.
    """
    def __init__(self, scenario):
        """Create the list of hazard levels from the scenario values"""
        self.scenario = scenario
        self.num_hazard_pts = \
            int(round((scenario.haz_param_max - scenario.haz_param_min) /
                      float(scenario.haz_param_step) + 1))

        self.hazard_intensity_vals = \
            np.linspace(scenario.haz_param_min, scenario.haz_param_max,
                        num=self.num_hazard_pts)

    def hazard_range(self):
        """A generator that can flexibly create hazard levels"""
        for hazard_intensity in self.hazard_intensity_vals:
            yield HazardLevel(self.scenario, hazard_intensity)


class HazardLevel(object):
    """A particular level of hazard in the range of hazard levels."""
    def __init__(self, scenario, hazard_intensity):
        """Only the intensity is currently used."""
        self.hazard_type = scenario.hazard_type
        self.intensity_measure_param = scenario.intensity_measure_param
        self.intensity_measure_unit = scenario.intensity_measure_unit
        if scenario.level_factor_raster:
            self.level_factor_raster = scenario.level_factor_raster
        else:
            self.level_factor_raster = None
        self.hazard_intensity = hazard_intensity

    def determine_intensity_at(self, location=None):
        if not location or (location.lon is np.NAN):
            return self.hazard_intensity
        else:
            # TODO Implement the loading of a hazard level factor raster
            if not self.level_factor_raster:
                raise RuntimeError("A location was given, but a location raster has not been configured")
            # use the lat long to offset into the NetCDF
            return self.level_factor_raster.variables['pga_factor'][location.lat, location.lon]*self.hazard_intensity

