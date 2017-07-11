import numpy as np


class HazardLevels(object):
    def __init__(self, sc):
        self.sc = sc
        self.num_hazard_pts = \
            int(round((sc.haz_param_max - sc.haz_param_min) /
                      float(sc.haz_param_step) + 1))

        self.hazard_intensity_vals = \
            np.linspace(sc.haz_param_min, sc.haz_param_max,
                        num=self.num_hazard_pts)

    def hazard_range(self):
        for hazard_intensity in self.hazard_intensity_vals:
            yield HazardLevel(self.sc, hazard_intensity)


class HazardLevel(object):
    def __init__(self, sc, hazard_intensity):
        self.intensity_measure_param = sc.intensity_measure_param
        self.intensity_measure_unit = sc.intensity_measure_unit
        self.hazard_intensity = hazard_intensity

