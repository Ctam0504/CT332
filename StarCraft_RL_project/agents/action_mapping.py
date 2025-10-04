import numpy as np
from pysc2.lib import actions as sc2_actions

# Action IDs
_NO_OP = sc2_actions.FUNCTIONS.no_op.id
_SELECT_ARMY = sc2_actions.FUNCTIONS.select_army.id
_MOVE_SCREEN = sc2_actions.FUNCTIONS.Move_screen.id
_SELECT_ALL = [0]

def map_beacon(non_spatial_action: int, spatial):
    """
    Ánh xạ action agent -> dict action cho MoveToBeacon.
    spatial: tuple hoặc list [x, y]
    """
    if isinstance(spatial, (int, float)):
        spatial = (0, 0)
    x, y = map(int, spatial)

    if non_spatial_action == 0:
        return {"function": _NO_OP, "args": []}
    elif non_spatial_action == 1:
        return {"function": _SELECT_ARMY, "args": [[_SELECT_ALL]]}
    elif non_spatial_action == 2:
        # args phải là [queued, [x, y]]
        return {"function": _MOVE_SCREEN,
                "args": [np.array([0], dtype=np.int32),
                         np.array([x, y], dtype=np.int32)]}
    else:
        return {"function": _NO_OP, "args": []}
