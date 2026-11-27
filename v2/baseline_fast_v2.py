import argparse
from pathlib import Path
import yaml
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0, shaping=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf),
            stream_metadata = { # All of this is part is optional
                "user": "v2-default", # choose your own username
                "env_id": rank, # environment identifier
                "color": "#447799", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
        if shaping:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from pokemon_game.reward_shaping import NavigationShaping
            env = NavigationShaping(env, shaping["graph"], env_conf["gb_path"],
                                    shaping["beta"], shaping["gamma"])
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    training_config = config["training"]
    reward_config = config["rewards"]

    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_path = Path(training_config["session_path"])
    sess_path.mkdir(parents=True, exist_ok=True)
    sess_id = sess_path.name
    with open(sess_path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': '../init.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25,
                'rewards': reward_config,
            }
    
    print(env_config)
    
    num_cpu = training_config["num_cpu"]
    seed = training_config["seed"]
    shaping = config.get("navigation_shaping")
    if shaping:
        shaping = {**shaping, "gamma": training_config["gamma"]}
    env = SubprocVecEnv([make_env(i, env_config, seed=seed, shaping=shaping) for i in range(num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="poke")
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)

    ppo_config = {
        "n_steps": training_config["n_steps"],
        "batch_size": training_config["batch_size"],
        "n_epochs": training_config["n_epochs"],
        "gamma": training_config["gamma"],
        "ent_coef": training_config["ent_coef"],
        "learning_rate": training_config["learning_rate"],
        "seed": training_config["seed"],
        "device": training_config["device"],
        "tensorboard_log": str(sess_path),
        "verbose": 1,
    }
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, **ppo_config)
    else:
        model = PPO("MultiInputPolicy", env, **ppo_config)
    
    print(model.policy)

    model.learn(
        total_timesteps=training_config["total_timesteps"],
        callback=CallbackList(callbacks),
        tb_log_name="poke_ppo",
        reset_num_timesteps=args.checkpoint is None,
    )

    if use_wandb_logging:
        run.finish()
