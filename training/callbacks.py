
import wandb
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode

# wandb.init(project='traffic_sim', entity='denbuz', reinit=True)


class MyCallbacks(DefaultCallbacks):
    'Callbacks for the traffic sim'

    def on_episode_end(self, *, worker, base_env, policies, episode: MultiAgentEpisode, env_index, **kwargs) -> None:
        # info = episode.last_info_for()  # Might use this later!
        # reward = episode.total_reward
        # wandb.log({'total_reward': reward})
        pass

    def on_train_result(self, *, trainer, result: dict, **kwargs) -> None:
        # result_floats = {}
        # for key in result:
        #     val = result[key]
        #     if isinstance(val, float) or isinstance(val, int):
        #         result_floats[key] = float(val)
        # wandb.log(result_floats)
        pass
