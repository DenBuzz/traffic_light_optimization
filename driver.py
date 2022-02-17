
import ray
import tensorflow as tf

import wandb
from training.configs import TUNE_CONFIG

tf.debugging.set_log_device_placement(True)
print(tf.config.list_physical_devices())

if __name__ == '__main__':

    config = TUNE_CONFIG
    # wandb.init(project='traffic_sim', entity='denbuz',
    #    config={'name': config['name']})

    analysis = ray.tune.run(**config)
    ray.shutdown()
