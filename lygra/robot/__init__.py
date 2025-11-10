# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lygra.robot.allegro import Allegro
from lygra.robot.shadow import Shadow
from lygra.robot.leap import Leap
from lygra.robot.dclaw import DClaw


def build_robot(name, urdf_path=None):
    if name == 'allegro':
        return Allegro(urdf_path=urdf_path)
    
    elif name == 'leap':
        return Leap(urdf_path=urdf_path)
    
    elif name == 'shadow':
        return Shadow(urdf_path=urdf_path)

    elif name == 'dclaw':
        return DClaw(urdf_path=urdf_path)
    
    else:
        assert False, f"Robot {name} undefined."
