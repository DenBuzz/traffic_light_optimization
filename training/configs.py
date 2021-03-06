

import os

from dijkstar import Graph
from gym.spaces import Box, Discrete
from ray import tune

from training.callbacks import MyCallbacks
from training.generate_graph import generate_graph
from training.platforms.road import Road
from training.platforms.traffic_env import TrafficEnv
from training.platforms.traffic_light import TrafficLight


def create_graph():

    tl_1 = TrafficLight(100, 100)
    tl_2 = TrafficLight(200, 100)
    tl_3 = TrafficLight(100, 200)
    tl_4 = TrafficLight(220, 180)
    tl_5 = TrafficLight(400, 200)
    tl_6 = TrafficLight(500, 200)
    tl_7 = TrafficLight(100, 300)
    tl_8 = TrafficLight(200, 300)
    tl_9 = TrafficLight(300, 300)
    tl_10 = TrafficLight(400, 300)
    tl_11 = TrafficLight(500, 300)

    rd_1 = Road(tl_1, tl_2, 'E', 30)
    rd_2 = Road(tl_2, tl_1, 'W', 30)

    rd_3 = Road(tl_1, tl_3, 'S', 30)
    rd_4 = Road(tl_3, tl_1, 'N', 30)

    rd_5 = Road(tl_2, tl_4, 'S', 30)
    rd_6 = Road(tl_4, tl_2, 'N', 30)

    rd_7 = Road(tl_3, tl_4, 'E', 25)
    rd_8 = Road(tl_4, tl_3, 'W', 25)

    rd_9 = Road(tl_4, tl_5, 'E', 40)
    rd_10 = Road(tl_5, tl_4, 'W', 40)

    rd_11 = Road(tl_5, tl_6, 'E', 25)
    rd_12 = Road(tl_6, tl_5, 'W', 25)

    rd_13 = Road(tl_3, tl_7, 'S', 25)
    rd_14 = Road(tl_7, tl_3, 'N', 25)

    rd_15 = Road(tl_4, tl_8, 'S', 25)
    rd_16 = Road(tl_8, tl_4, 'N', 25)

    rd_17 = Road(tl_5, tl_10, 'S', 25)
    rd_18 = Road(tl_10, tl_5, 'N', 25)

    rd_19 = Road(tl_6, tl_11, 'S', 25)
    rd_20 = Road(tl_11, tl_6, 'N', 25)

    rd_21 = Road(tl_7, tl_8, 'E', 25)
    rd_22 = Road(tl_8, tl_7, 'W', 25)

    rd_23 = Road(tl_8, tl_9, 'E', 25)
    rd_24 = Road(tl_9, tl_8, 'W', 25)

    rd_25 = Road(tl_9, tl_10, 'E', 25)
    rd_26 = Road(tl_10, tl_9, 'W', 25)

    rd_27 = Road(tl_10, tl_11, 'E', 25)
    rd_28 = Road(tl_11, tl_10, 'W', 25)

    roads = [rd_1,
             rd_2,
             rd_3,
             rd_4,
             rd_5,
             rd_6,
             rd_7,
             rd_8,
             rd_9,
             rd_10,
             rd_11,
             rd_12,
             rd_13,
             rd_14,
             rd_15,
             rd_16,
             rd_17,
             rd_18,
             rd_19,
             rd_20,
             rd_21,
             rd_22,
             rd_23,
             rd_24,
             rd_25,
             rd_26,
             rd_27,
             rd_28
             ]

    graph = Graph(undirected=False)

    for rd in roads:
        graph.add_edge(rd.start, rd.end, rd)
        rd.start.add_outgoing_road(rd)
        rd.end.add_incoming_road(rd)
    
    for i, light in enumerate(graph.get_data()):
        light.light_id = f'traffic_light_{i}'

    return graph


graph = generate_graph(grid_size=(5, 4), lights=10)

policies = {f'traffic_light_{i}': (None, Box(low=-2, high=1000, shape=(16,), dtype=float), Discrete(4), {}) for i in range(graph.node_count)}


GENERAL_CONFIG = {
    'env': TrafficEnv,
    'env_config': {
        # Either pass a graph or args for generate graph.
        'graph': graph,#{'grid_size': (5, 4), 'lights': 14},
        'steps_per_action': 1,
        'sim_dt': 1,
        'sim_random_car_probability': 0.8,
        'episode_length': 10000,
    },
    # 'multiagent': {
    #     'policies': {
    #         'traffic_light': (None, Box(low=-2, high=1000, shape=(16,), dtype=float), Discrete(4), {})
    #     },
    #     'policy_mapping_fn': lambda _: 'traffic_light',
    # },
    'multiagent':{
        'policies': policies,
        'policy_mapping_fn': lambda x: x,
    },
    'callbacks': MyCallbacks,
    'num_gpus': 1,
    'num_workers': 4,
    'render_env': False,
    'output': './training_data',
    'num_cpus_per_worker': 1,
    'metrics_smoothing_episodes': 10,
}

# PPO_CONFIG = {
#     'model': {
#         'use_lstm': True,
#         'lstm_use_prev_action': True,
#         'lstm_cell_size': 64,
#         'max_seq_len': 20,
#     }
# }

# APEX_CONFIG = {
#     'model': {
#         'buffer_size':100000        
#     }
# }

# GENERAL_CONFIG.update(APEX_CONFIG)

TUNE_CONFIG = {
    'name': 'multi_policy',
    'run_or_experiment': 'PPO',
    'stop': {'training_iteration': 100},
    'config': GENERAL_CONFIG,
    'local_dir': './tune_data',
    'checkpoint_freq': 1,
    'checkpoint_at_end': True,
    'checkpoint_score_attr': 'episode_reward_mean',
    'keep_checkpoints_num': 5,
}

