"""
This script implements a simulation of cluster running two apps with their different services.
The cluster attempts to represent a spatially separated system of devices, the components are:
- One sensor device
- Two cloud devices
- Two actuators devices

The bandwidth and delay of each link will be different to simulate spatially separated devices.
"""

import random

from yafs.core import Sim
from yafs.application import Application, Message

from yafs.population import *
from yafs.topology import Topology

from simpleSelection import MinimunPath
from simplePlacement import CloudPlacement
from yafs.stats import Stats
from yafs.distribution import deterministicDistribution
from yafs.utils import fractional_selectivity
import time
import numpy as np

RANDOM_SEED = 1


def create_application(name):
    """
    This function creates the application services and messages(requests).
    In this simulation both apps will have the same logic. Thus, the modules will be set to be static.

    The services will be:
        - Source (sensor)
        - Module (cloud service)
        - Sink (actuator)
    The sensor will always send the message to the service and the service will send the message to the actuator.

    name: name of the application to be created
    """

    # Create application object with name App1
    a = Application(name=name)

    # Define the modules of the application and their Direction of communication. Messages direction
    a.set_modules([{"Sensor": {"Type": Application.TYPE_SOURCE}},
                   {"Service": {"RAM": 10, "Type": Application.TYPE_MODULE}},
                   {"Actuator": {"Type": Application.TYPE_SINK}}
                   ])

    # Sensor ==> M.A ==> ServiceA
    # ServiceA ==> M.B ==> Actuator
    m_a = Message("M.A", "Sensor", "Service", instructions=30 * 10 ^ 6, bytes=1000)
    m_b = Message("M.B", "Service", "Actuator", instructions=30 * 10 ^ 6, bytes=1000)

    # Defining which messages will be dynamically generated, the generation is controlled by Population algorithm
    a.add_source_messages(m_a)

    # MODULE SERVICES
    a.add_service_module("Service", m_a, m_b, fractional_selectivity, threshold=1.0)

    return a


def create_json_topology():
    """
    This function returns the topology of the cluster in the form of a dictionary
    The cluster in this test will be formed by:
        - A sensor device
        - Two module or cloud device
        - Two actuators devices
    """

    # Instantiate the topology with JSON structure.
    # "entity" will contain nodes of the cluster
    # "link" will contain network connexions between nodes
    topology_json = {"entity": [],
                     "link": []}

    sensor_dev = {"id": 0, "model": "sensor-device", "IPT": 100 * 10 ^ 6, "RAM": 4000, "COST": 3, "WATT": 40.0}
    cloud_dev_1 = {"id": 1, "model": "cloud", "mytag": "cloud1", "IPT": 5000 * 10 ^ 6, "RAM": 40000, "COST": 3, "WATT": 20.0}
    cloud_dev_2 = {"id": 2, "model": "cloud", "mytag": "cloud2", "IPT": 5000 * 10 ^ 6, "RAM": 40000, "COST": 3, "WATT": 20.0}
    actuator_dev_1 = {"id": 3, "model": "actuator-device-1", "IPT": 100 * 10 ^ 6, "RAM": 4000, "COST": 3, "WATT": 40.0}
    actuator_dev_2 = {"id": 4, "model": "actuator-device-2", "IPT": 100 * 10 ^ 6, "RAM": 4000, "COST": 3, "WATT": 40.0}

    link1 = {"s": 0, "d": 1, "BW": 1, "PR": 2}
    link2 = {"s": 0, "d": 2, "BW": 1, "PR": 4}
    link3 = {"s": 1, "d": 3, "BW": 1, "PR": 2}
    link4 = {"s": 1, "d": 4, "BW": 1, "PR": 10}
    link5 = {"s": 2, "d": 4, "BW": 1, "PR": 6}

    topology_json["entity"].append(sensor_dev)
    topology_json["entity"].append(cloud_dev_1)
    topology_json["entity"].append(cloud_dev_2)
    topology_json["entity"].append(actuator_dev_1)
    topology_json["entity"].append(actuator_dev_2)

    topology_json["link"].append(link1)
    topology_json["link"].append(link2)
    topology_json["link"].append(link3)
    topology_json["link"].append(link4)
    topology_json["link"].append(link5)

    return topology_json


def main(simulated_time):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_apps = 2

    """
    TOPOLOGY from a json
    """
    t = Topology()
    t_json = create_json_topology()
    t.load(t_json)
    t.write("network.gexf")

    """
    APPLICATIONS and POPULATION
    """
    apps = [create_application("app_" + str(idx + 1)) for idx in range(n_apps)]
    pops = [Statical("Statical_" + str(idx + 1)) for idx in range(n_apps)]

    """
    PLACEMENT algorithm
    """
    cloud_tags = ["cloud1", "cloud2"]
    placements = []
    for idx in range(n_apps):
        placement = CloudPlacement("onCloud" + str(idx), tag=cloud_tags[idx])
        placement.scaleService({"Service": 1})
        placements.append(placement)

    """
    POPULATION algorithm
    """
    # In ifogsim, during the creation of the application, the Sensors are assigned to the topology, in this case no. As mentioned, YAFS differentiates the adaptive sensors and their topological assignment.
    # In their case, the use a statical assignment.
    # For each type of sink modules we set a deployment on some type of devices
    # A control sink consists on:
    #  args:
    #     model (str): identifies the device or devices where the sink is linked
    #     number (int): quantity of sinks linked in each device
    #     module (str): identifies the module from the app who receives the messages
    for idx, pop in enumerate(pops):
        # Distribution to generate messages
        d_distribution = deterministicDistribution(name="Deterministic", time=100)
        pop.set_sink_control({"model": "actuator-device-" + str(idx+1), "number": 1, "module": apps[idx].get_sink_modules()[0]})
        pop.set_src_control({"model": "sensor-device", "number": 1, "message": apps[idx].get_message("M.A"),
                             "distribution": d_distribution})

    """
    SELECTOR algorithm
    """
    # Their "selector" is actually the shortest way, there is not type of orchestration algorithm.
    # This implementation is already created in selector.class,called: First_ShortestPath
    selector_path = MinimunPath()

    """
    SIMULATION ENGINE
    """
    stop_time = simulated_time
    s = Sim(t, default_results_path="Results")

    for idx, app in enumerate(apps):
        s.deploy_app(app, placements[idx], pops[idx], selector_path)

    s.run(stop_time, show_progress_monitor=False)

    s.draw_allocated_topology()  # for debugging


if __name__ == '__main__':
    import logging.config
    import os

    logging.config.fileConfig(os.getcwd() + '/logging.ini')

    start_time = time.time()
    main(simulated_time=10000)

    print("\n--- %s seconds ---" % (time.time() - start_time))

    ### Finally, you can analyse the results:
    # print "-"*20
    # print "Results:"
    # print "-" * 20
    m = Stats(defaultPath="Results")  # Same name of the results
    time_loops = [["M.A", "M.B"]]
    m.showResults2(10000, time_loops=time_loops)
    # print "\t- Network saturation -"
    # print "\t\tAverage waiting messages : %i" % m.average_messages_not_transmitted()
    # print "\t\tPeak of waiting messages : %i" % m.peak_messages_not_transmitted()PartitionILPPlacement
    # print "\t\tTOTAL messages not transmitted: %i" % m.messages_not_transmitted()
