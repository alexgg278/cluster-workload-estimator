"""
1. The goal of this test is to reproduce similar results but by adding noise to the memory utilization of the nodes.
This can be easily changed by adding a standard deviation for the distribution of memory usage with mean = bytes

2. There are 2 types of noise the one that at every time-step the values differ, and the one that at every cycle of a
job in a node the memory utilization is different but constant in this interval

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

from parameters import Parameters

RANDOM_SEED = None


def create_application(name, modules, messages, transmissions):
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
    a.set_modules(modules)

    # Create the messages
    messages_list = []
    for message in messages:
        messages_list.append(Message(message["name"], message["src"], message["dst"], message["instructions"], message["bytes"], message["std"]))

    messages_dict = {}
    for message in messages_list:
        messages_dict[message.name] = message

    # Defining which messages will be dynamically generated, the generation is controlled by Population algorithm
    for idx, message in enumerate(messages):
        if message["pop"]:
            a.add_source_messages(messages_list[idx])

    # MODULE SERVICES
    for idx, module in enumerate(modules):
        # Actuators can also generate ACK messages
        if module.values()[0]["Type"] == Application.TYPE_MODULE:
            for service in transmissions.keys():
                if service == module.keys()[0]:
                    for t in transmissions[service]:
                        a.add_service_module(service, messages_dict[t["in"]], messages_dict[t["out"]], fractional_selectivity, threshold=t["threshold"])

    return a

def create_json_topology():
    """
    This function returns the topology of the cluster in the form of a dictionary
    The cluster in this test will be formed by:
        - A sensor device
        - Two module or cloud device
        - Two actuators devices
    """

    sensor_dev_1 = {"id": 0, "model": "sensor-device-1", "IPT": 10 * 10 ^ 6, "RAM": 1000, "COST": 1, "WATT": 40.0}
    sensor_dev_2 = {"id": 1, "model": "sensor-device-2", "IPT": 10 * 10 ^ 6, "RAM": 1000, "COST": 1, "WATT": 40.0}
    cloud_dev_1 = {"id": 2, "model": "cloud", "mytag": "cloud1", "IPT": 500 * 10 ^ 6, "RAM": 10000, "COST": 3, "WATT": 20.0}
    cloud_dev_2 = {"id": 3, "model": "cloud", "mytag": "cloud2", "IPT": 1000 * 10 ^ 6, "RAM": 40000, "COST": 3, "WATT": 20.0}
    cloud_dev_3 = {"id": 4, "model": "cloud", "mytag": "cloud3", "IPT": 500 * 10 ^ 6, "RAM": 10000, "COST": 3, "WATT": 20.0}
    actuator_dev_1 = {"id": 5, "model": "actuator-device-1", "IPT": 100 * 10 ^ 6, "RAM": 2000, "COST": 2, "WATT": 40.0}
    actuator_dev_2 = {"id": 6, "model": "actuator-device-2", "IPT": 100 * 10 ^ 6, "RAM": 2000, "COST": 2, "WATT": 40.0}

    link1 = {"s": 0, "d": 2, "BW": 0.5, "PR": 8}
    link2 = {"s": 1, "d": 3, "BW": 0.5, "PR": 8}
    link3 = {"s": 1, "d": 4, "BW": 0.5, "PR": 8}
    link4 = {"s": 2, "d": 3, "BW": 2, "PR": 1}
    link5 = {"s": 3, "d": 2, "BW": 2, "PR": 1}
    link6 = {"s": 3, "d": 1, "BW": 0.5, "PR": 8}
    link7 = {"s": 3, "d": 5, "BW": 4, "PR": 0.5}
    link8 = {"s": 4, "d": 6, "BW": 2, "PR": 1}
    link9 = {"s": 5, "d": 3, "BW": 4, "PR": 0.5}
    link10 = {"s": 6, "d": 4, "BW": 2, "PR": 1}

    # Instantiate the topology with JSON structure.
    # "entity" will contain nodes of the cluster
    # "link" will contain network connexions between nodes
    topology_json = {"entity": [sensor_dev_1, sensor_dev_2, cloud_dev_1, cloud_dev_2, cloud_dev_3, actuator_dev_1, actuator_dev_2],
                     "link": [link1, link2, link3, link4, link5, link6, link7, link8, link9, link10]}

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
    modules = {"app_1": [{"Sensor1": {"Type": Application.TYPE_SOURCE}},
                         {"Sensor2": {"Type": Application.TYPE_SOURCE}},
                         {"Service1": {"RAM": 200, "Type": Application.TYPE_MODULE}},
                         {"Service2": {"RAM": 100, "Type": Application.TYPE_MODULE}},
                         {"Actuator1": {"Type": Application.TYPE_SINK}}],
               "app_2": [{"Sensor1": {"Type": Application.TYPE_SOURCE}},
                         {"Service1": {"RAM": 100, "Type": Application.TYPE_MODULE}},
                         {"Service2": {"RAM": 200, "Type": Application.TYPE_MODULE}},
                         {"Actuator1": {"Type": Application.TYPE_SINK}},
                         {"Actuator2": {"Type": Application.TYPE_SINK}}]
               }

    messages = {"app_1": [{"name": "m_s1", "src": "Sensor1", "dst": "Service1", "instructions": 10000 * 10 ^ 6, "bytes": 5000, "pop": True, "std": None},
                          {"name": "m_s2", "src": "Sensor2", "dst": "Service2", "instructions": 25000 * 10 ^ 6, "bytes": 9500, "pop": True, "std": None},
                          {"name": "m_ack_s2", "src": "Service2", "dst": "Sensor2", "instructions": 6500 * 10 ^ 6, "bytes": 500, "pop": False, "std": None},
                          {"name": "m_c1", "src": "Service1", "dst": "Service2", "instructions": 30000 * 10 ^ 6, "bytes": 15000, "pop": False, "std": None},
                          {"name": "m_ack_c2", "src": "Service2", "dst": "Service2", "instructions": 10000 * 10 ^ 6, "bytes": 20000, "pop": False, "std": None},
                          {"name": "m_ack_c1", "src": "Service2", "dst": "Service1", "instructions": 12000 * 10 ^ 6,
                           "bytes": 5000, "pop": False, "std": None},
                          {"name": "m_c2", "src": "Service1", "dst": "Service2", "instructions": 25000 * 10 ^ 6,
                           "bytes": 65000, "pop": False, "std": None},
                          {"name": "m_a3", "src": "Service2", "dst": "Actuator1", "instructions": 2500 * 10 ^ 6,
                           "bytes": 2000, "pop": False, "std": None},
                          {"name": "m_a1", "src": "Service2", "dst": "Actuator1", "instructions": 10000 * 10 ^ 6, "bytes": 4400, "pop": False, "std": None},
                          {"name": "m_a2", "src": "Service2", "dst": "Actuator1", "instructions": 12000 * 10 ^ 6, "bytes": 5600, "pop": False, "std": None}],
                "app_2": [{"name": "m_s1", "src": "Sensor1", "dst": "Service1", "instructions": 35000 * 10 ^ 6, "bytes": 16000, "pop": True, "std": None},
                          {"name": "m_s2", "src": "Sensor1", "dst": "Service2", "instructions": 25000 * 10 ^ 6, "bytes": 12200, "pop": True, "std": None},
                          {"name": "m_a1", "src": "Service1", "dst": "Actuator1", "instructions": 3000 * 10 ^ 6, "bytes": 3400, "pop": False, "std": None},
                          {"name": "m_a2", "src": "Service2", "dst": "Actuator2", "instructions": 3000 * 10 ^ 6, "bytes": 2100, "pop": False, "std": None}]
                }

    transmissions = {"app_1": {"Service1": [{"in": "m_s1", "out": "m_c1", "threshold": 1},
                                            {"in": "m_ack_c1", "out": "m_c2", "threshold": 1}],
                               "Service2": [{"in": "m_c1", "out": "m_ack_c2", "threshold": 0.1},
                                            {"in": "m_ack_c2", "out": "m_ack_c1", "threshold": 1},
                                            {"in": "m_s2", "out": "m_ack_s2", "threshold": 1},
                                            {"in": "m_c1", "out": "m_a1", "threshold": 1},
                                            {"in": "m_c2", "out": "m_a3", "threshold": 1},
                                            {"in": "m_s2", "out": "m_a2", "threshold": 1}]},
                     "app_2": {"Service1": [{"in": "m_s1", "out": "m_a1", "threshold": 1}],
                               "Service2": [{"in": "m_s2", "out": "m_a2", "threshold": 1}]}}

    apps = [create_application("app_" + str(idx + 1),
                               modules["app_" + str(idx + 1)],
                               messages["app_" + str(idx + 1)],
                               transmissions["app_" + str(idx + 1)])
            for idx in range(n_apps)]

    """
    PLACEMENT algorithm
    """
    cloud_tags = [["cloud1", "cloud2"], ["cloud2", "cloud3"]]
    placements = []

    placement1 = CloudPlacement("onCloud1", tag=cloud_tags[0])
    placement1.scaleService({"Service1": 1, "Service2": 1})
    placements.append(placement1)

    placement2 = CloudPlacement("onCloud2", tag=cloud_tags[1])
    placement2.scaleService({"Service1": 1, "Service2": 1})
    placements.append(placement2)

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
    pops = []
    # Distribution to generate messages
    d_distribution = deterministicDistribution(name="Deterministic", time=75)
    pop1 = Statical("Statical_1")
    pop1.set_sink_control({"model": "actuator-device-1", "number": 1, "module": "Actuator1"})
    pop1.set_src_control({"model": "sensor-device-1", "number": 1, "message": apps[0].get_message("m_s1"),
                         "distribution": d_distribution})
    pop1.set_src_control({"model": "sensor-device-2", "number": 1, "message": apps[0].get_message("m_s2"),
                          "distribution": d_distribution})

    pop2 = Statical("Statical_2")
    pop2.set_sink_control({"model": "actuator-device-2", "number": 1, "module": "Actuator2"})
    pop2.set_sink_control({"model": "actuator-device-1", "number": 1, "module": "Actuator1"})
    pop2.set_src_control({"model": "sensor-device-2", "number": 1, "message": apps[1].get_message("m_s2"),
                          "distribution": d_distribution})
    pop2.set_src_control({"model": "sensor-device-2", "number": 1, "message": apps[1].get_message("m_s1"),
                          "distribution": d_distribution})

    pops = [pop1, pop2]

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

    param = Parameters()

    logging.config.fileConfig(os.getcwd() + '/logging.ini')

    start_time = time.time()
    main(simulated_time=param.simulation_time)

    print("\n--- %s seconds ---" % (time.time() - start_time))

    ### Finally, you can analyse the results:
    # print "-"*20
    # print "Results:"
    # print "-" * 20
    m = Stats(defaultPath="Results")  # Same name of the results
    time_loops = [["M.A_1", "M.B_1", "M.A_2", "M.B_2"]]
    m.showResults2(param.simulation_time, time_loops=time_loops)
    # print "\t- Network saturation -"
    # print "\t\tAverage waiting messages : %i" % m.average_messages_not_transmitted()
    # print "\t\tPeak of waiting messages : %i" % m.peak_messages_not_transmitted()PartitionILPPlacement
    # print "\t\tTOTAL messages not transmitted: %i" % m.messages_not_transmitted()
