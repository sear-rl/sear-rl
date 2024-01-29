from d4rl.kitchen.kitchen_envs import KitchenBase
from gym.envs.registration import register

register(
    id='kitchen-microwave-v0',
    entry_point='sear.environments.kitchen_custom_envs:KitchenMicrowaveV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url': None
    })

register(
    id='kitchen-kettle-v0',
    entry_point='sear.environments.kitchen_custom_envs:KitchenKettleV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url': None
    })

register(
    id='kitchen-light-v0',
    entry_point='sear.environments.kitchen_custom_envs:KitchenLightV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url': None
    })

register(
    id='kitchen-slider-v0',
    entry_point='sear.environments.kitchen_custom_envs:KitchenSliderV0',
    max_episode_steps=280,
    kwargs={
        'ref_min_score': 0.0,
        'ref_max_score': 1.0,
        'dataset_url': None
    })


class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ['microwave']


class KitchenKettleV0(KitchenBase):
    TASK_ELEMENTS = ['kettle']


class KitchenLightV0(KitchenBase):
    TASK_ELEMENTS = ['light switch']


class KitchenSliderV0(KitchenBase):
    TASK_ELEMENTS = ['slide cabinet']
