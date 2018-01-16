import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register (
    id = "IceGameEnv-v3",
    entry_point="gym_icegame.envs:IceGameEnv",
    kwargs={
        "L" : 32,
        "kT" : 0.0001,
        "J" : 1.0,
        "is_cont": False,
    },
    )