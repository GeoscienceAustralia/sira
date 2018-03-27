import unittest as ut
from netCDF4 import Dataset
import numpy as np

from sifra.scenario import Scenario
from sifra.modelling.hazard import Hazards, HazardLevel
from sifra.modelling.component import Location
from sifra.configuration import Configuration

config_file = './simulation_setup/test_scenario_ps_coal.json'


class TestIngestResponseModel(ut.TestCase):
    def test_hazard_levels(self):

        configuration_file_path = './simulation_setup/test_scenario_ps_coal.json'
        config = Configuration(configuration_file_path)
        hazard_level_it = Hazards(config)

        hazard_levels = list(hazard_level_it.hazard_scenario_list())

        self.assertTrue(len(hazard_levels) > 4)

        location = Location(np.NAN, np.NAN, np.NAN)

        for hazard_level in hazard_levels:
            self.assertTrue(hazard_level.determine_intensity_at(location) > 0)

    def test_hazard_raster(self):
        configuration_file_path = './simulation_setup/test_scenario_ps_coal.json'
        config = Configuration(configuration_file_path)
        scenario = Scenario(config)

        test_raster = Dataset("test_raster.nc", "w", format="NETCDF4")
        lats = np.arange(-90, 91, 1.0)
        lons = np.arange(-180, 180, 1.0)
        lat = test_raster.createDimension("lat", len(lats))
        lon = test_raster.createDimension("lon", len(lons))
        latitudes = test_raster.createVariable("lat", "f4", ("lat",))
        longitudes = test_raster.createVariable("lon", "f4", ("lon",))
        pga_factor = test_raster.createVariable("pga_factor", "f4", ("lat", "lon",), fill_value=False)
        test_raster.description = "unit test raster"
        test_raster.source = "netCDF4 python module tutorial"
        latitudes.units = "degrees north"
        longitudes.units = "degrees east"
        latitudes[:] = lats
        longitudes[:] = lons
        pga_factor[:, :] = np.ones((len(lats), len(lons)))

        scenario.level_factor_raster = test_raster

        hazard_levels = Hazards(config)

        hazard_levels = list(hazard_levels.hazard_scenario_list())

        self.assertTrue(len(hazard_levels) > 4)

        location = Location(-28.0, 135.0, 0)

        for hazard_level in hazard_levels:
            self.assertTrue(hazard_level.determine_intensity_at(location) > 0)


if __name__ == '__main__':
    ut.main()
