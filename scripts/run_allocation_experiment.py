# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example of training a policy with OmniSafe."""

import argparse
import sys
sys.path.append('../')

import omnisafe
from omnisafe.utils.tools import custom_cfgs_to_dict, update_dict

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algo',
        type=str,
        metavar='ALGO',
        default='PETS',
        help='algorithm to train',
        choices=omnisafe.ALGORITHMS['all'],
    )
    parser.add_argument(
        '--env-id',
        type=str,
        metavar='ENV',
        default='Allocation-v0',
        help='the name of test environment',
    )
    parser.add_argument(
        '--parallel',
        default=1,
        type=int,
        metavar='N',
        help='number of paralleled progress for calculations.',
    )
    parser.add_argument(
        '--vector-env-nums',
        type=int,
        default=1,
        metavar='VECTOR-ENV',
        help='number of vector envs to use for training',
    )
    parser.add_argument(
        '--torch-threads',
        type=int,
        default=16,
        metavar='THREADS',
        help='number of threads to use for torch',
    )
    parser.add_argument(
        '--use_nde_r',
        type=bool,
        default=False,
        metavar='NDE_R',
        help='whether to use NDE(R)',
    )
    parser.add_argument(
        '--use_nde_s',
        type=bool,
        default=False,
        metavar='NDE_S',
        help='whether to use NDE(S)',
    )
    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        metavar='MODE',
        help='0: PETS; 1: InsightFair; 2: Fair-A; 3: Fair-S',
    )
    args, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))

    custom_cfgs = {}
    for k, v in unparsed_args.items():
        update_dict(custom_cfgs, custom_cfgs_to_dict(k, v))
    
    device_id = 0
    device = f'cuda:{device_id}'
    update_dict(custom_cfgs, {'train_cfgs': {'device': device}})

    if args.algo == 'PETS':
        update_dict(custom_cfgs, {'algo_cfgs': {'use_nde_r': args.use_nde_r, 'use_nde_s': args.use_nde_s}})
        print("Using NDE(R): ", args.use_nde_r, "; Using NDE(S): ", args.use_nde_s)
        if not args.use_nde_r and not args.use_nde_s:
            assert args.mode != 1, "If not using NDE(R) or NDE(S), then mode should not be 1 (InsightFair)."
        update_dict(custom_cfgs, {'planner_cfgs': {'mode': args.mode}})
        print("Using mode: ", args.mode, "; 0: PETS; 1: InsightFair; 2: Fair-A; 3: Fair-S")

    vars(args).pop('use_nde_r', None)
    vars(args).pop('use_nde_s', None)
    vars(args).pop('mode', None)

    update_dict(custom_cfgs, {
        'env_cfgs': {
            'alpha1': 0.0, 'alpha2': 0.05,
            'beta1': 0.0, 'beta2': 0.05,
            'rate1': 6.0, 'rate2': 6.0
        }
    })

    agent = omnisafe.Agent(
        args.algo,
        args.env_id,
        train_terminal_cfgs=vars(args),
        custom_cfgs=custom_cfgs,
    )
    agent.learn()
