PITCH_TYPES = [
    '4-Seam Fastball',
    'Slider',
    'Changeup',
    'Curveball',
    'Cutter',
    'Sinker',
    'Splitter',
    'Knuckle Curve',
    'Sweeper'
]

STRIKE_ZONE = {
    'left': -0.71,
    'right': 0.71,
    'bottom': 1.5,
    'top': 3.5
}
MOVEMENT_SCALE = 0.5
MOVEMENT_THRESHOLD = 0.05
BB_TYPES = ['fly_ball', 'ground_ball', 'line_drive', 'popup']
PITCH_PRESETS = {
    '4-Seam Fastball': {
        'R': {'release_speed': 94.52, 'pfx_x': -0.62, 'pfx_z': 1.33},
        'L': {'release_speed': 93.02, 'pfx_x': 0.65, 'pfx_z': 1.32},
    },
    'Slider': {
        'R': {'release_speed': 85.78, 'pfx_x':  0.41, 'pfx_z':  -0.14},
        'L': {'release_speed': 84.56, 'pfx_x':  -0.40, 'pfx_z':  -0.15},
    },
    'Changeup': {
        'R': {'release_speed': 86.06, 'pfx_x': -1.18, 'pfx_z':  0.44},
        'L': {'release_speed': 84.08, 'pfx_x': 1.18, 'pfx_z':  0.54},
    },
    'Curveball': {
        'R': {'release_speed': 79.41, 'pfx_x':  0.80, 'pfx_z': -0.83},
        'L': {'release_speed': 78.12, 'pfx_x':  -0.72, 'pfx_z': -0.79},
    },
    'Cutter': {
        'R': {'release_speed': 89.85, 'pfx_x':  0.21, 'pfx_z':  0.68},
        'L': {'release_speed': 87.37, 'pfx_x':  -0.15, 'pfx_z':  0.62},
    },
    'Sinker': {
        'R': {'release_speed': 93.66, 'pfx_x': -1.24, 'pfx_z':  0.65},
        'L': {'release_speed': 92.65, 'pfx_x': 1.26, 'pfx_z':  0.71},
    },
    'Splitter': {
        'R': {'release_speed': 86.51, 'pfx_x': -0.92, 'pfx_z':  0.27},
        'L': {'release_speed': 84.28, 'pfx_x': 0.76, 'pfx_z':  0.47},
    },
    'Knuckle Curve': {
        'R': {'release_speed': 82.10, 'pfx_x':  0.65, 'pfx_z': -0.89},
        'L': {'release_speed': 80.21, 'pfx_x':  -0.26, 'pfx_z': -0.58},
    },
    'Sweeper': {
        'R': {'release_speed': 82.44, 'pfx_x':  1.17, 'pfx_z':  0.10},
        'L': {'release_speed': 80.13, 'pfx_x':  -1.17, 'pfx_z':  0.09},
    },
}

LAUNCH_SPEED_ANGLE_LABELS = {
    1: 'Weak',
    2: 'Topped',
    3: 'Under',
    4: 'Flare / Burner',
    5: 'Solid Contact',
    6: 'Barrel'
}

MODEL_A_COLUMNS = [
    'pitch_name', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'pfx_x', 'pfx_z', 'balls', 'strikes', 'stand', 'p_throws',
    'release_extension', 'arm_angle', 'outs_when_up'
]

MODEL_B_COLUMNS = [
    'pitch_name', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate',
    'pfx_x', 'pfx_z', 'balls', 'strikes', 'stand', 'p_throws',
    'release_extension', 'arm_angle', 'outs_when_up', 'launch_speed_angle',
    'hc_x', 'hc_y', 'bb_type'
]
