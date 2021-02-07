"""
    This type of algorithm have two obligatory functions:

        *initial_allocation*: invoked at the start of the simulation

        *run* invoked according to the assigned temporal distribution.

"""

from yafs.placement import Placement


class CloudPlacement(Placement):
    """
    This implementation locates the services of the application in the cheapest cloud regardless of where the sources or sinks are located.

    It only runs once, in the initialization.

    """
    def __init__(self, name, tag=""):
        Placement.__init__(self, name)
        self.tag = tag

    def initial_allocation(self, sim, app_name):

        app = sim.apps[app_name]
        services = app.services
        services_list = services.keys()
        services_list.reverse()

        for idx, module in enumerate(services_list):
            if module in self.scaleServices:
                value = {"mytag": self.tag[idx]}
                id_cluster = sim.topology.find_IDs(value)
                for rep in range(0, self.scaleServices[module]):
                    idDES = sim.deploy_module(app_name,module,services[module],id_cluster)

    #end function




