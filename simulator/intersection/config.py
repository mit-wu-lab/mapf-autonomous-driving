import numpy as np
import argparse
from util import Path

render_images = False

seed = 0
lane_width = 4.5
car_width = 2
car_length = 5
lane_length = 250

max_speed = 13
right_speed = 4.5
left_speed = 6.5
u_speed = 3

init_speed = 5
max_accel = 2.6
max_decel = 4.5

parser = argparse.ArgumentParser()
parser.add_argument('run_dir',type=Path)
args = parser.parse_args()
run_dir = args.run_dir
config = (run_dir / 'config.yaml').load()
globals().update(config)
assert render or not render_images