from gym import register

STEPS = 30

register(
    id=f"PmtgTest-Vanilla-v0",
    entry_point="pmtg_test.envs:PmtgEnv",
    kwargs={"steps": STEPS,"with_timer":False, "with_pmtg":False},
    max_episode_steps=STEPS,
)

register(
    id=f"PmtgTest-VanillaTimer-v0",
    entry_point="pmtg_test.envs:PmtgEnv",
    kwargs={"steps": STEPS,"with_timer":True, "with_pmtg":False},
    max_episode_steps=STEPS,
)

register(
    id=f"PmtgTest-Pmtg-v0",
    entry_point="pmtg_test.envs:PmtgEnv",
    kwargs={"steps": STEPS,"with_timer":True, "with_pmtg":True},
    max_episode_steps=STEPS,
)


