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
        self.hazard_range = np.linspace(configuration.PGA_MIN,
                                                 configuration.PGA_MAX,
                                                 num=self.num_hazard_pts)

        # self.hazard_range = [HazardLevel(configuration, hazard_intensity)
        #                      for hazard_intensity in self.hazard_intensity_vals]

        self.intensity_measure_param = configuration.INTENSITY_MEASURE_PARAM
        self.intensity_measure_unit = configuration.INTENSITY_MEASURE_UNIT

        self.hazard_intensity_str = [('%0.3f' % np.float(x)) for x in self.hazard_range]



# class HazardLevel(object):
#     """A particular level of hazard in the range of hazard levels."""
#     def __init__(self, configuration, hazard_intensity):
#         """Only the intensity is currently used."""
#         self.hazard_type = configuration.HAZARD_TYPE
#
#         if configuration.HAZARD_RASTER:
#             self.level_factor_raster = configuration.HAZARD_RASTER
#         else:
#             self.level_factor_raster = None
#         self.hazard_intensity = hazard_intensity
#
#     def determine_intensity_at(self):
#             return self.hazard_intensity

