

class ExposableInfrastructure(object):
    """
    In design stage, and not currently implemented
    The intention is to implement the behavioural logic for the
    infrastructure under this class.
    """

    def __init__(self, component):
        self.component = component
        self.hazard_levels = None

    def expose_to(self, hazard_level, scenario):
        pass

    def get_damage_state(self, hazard_level):
        pass

    def get_economic_loss(self, hazard_level):
        pass

    def get_functionality(self, hazard_level):
        pass

