from setuptools import setup

setup(name="pmtg_test", version="1.0", install_requires=["tqdm", "numpy", "matplotlib", "gym",])

# python main.py --env-name "PmtgTest-Vanilla-v0" --custom-gym pmtg_test --num-env-steps 10e5 --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 200 --num-mini-batch 32 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --wandb pmtg-test-v1 --frame-stacc 1 --seed 1
