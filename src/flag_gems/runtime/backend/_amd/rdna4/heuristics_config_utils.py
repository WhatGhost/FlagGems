# Copyright 2026 FlagOS Contributors
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

"""RDNA4 heuristics for the non-inner reduction kernels.

`mean_non_inner` and `argmax_non_inner` are the only keys `_amd` never defines,
so `configs_loader` falls back to the `_nvidia` table for them. Both tables
drive the same kernel shape -- grid `(M, cdiv(K, TILE_K))`, each workgroup
looping over the reduction axis N in TILE_N steps -- but `_nvidia`'s constants
were fitted on a different machine and cost gfx1201 more than half its
throughput on some shapes:

* `mean_heur_tile_k` caps TILE_K at `_MAX_TILE_N_PER_ROW // _MIN_TILE_N`, that
  is 64, whatever K is. At K=512 each workgroup then reads 64 contiguous
  bfloat16 -- 128 bytes -- out of every 1024-byte row, and `[64,512,512]` runs
  at 0.4x torch. `sum_dim`, which is the same kernel body reading `_amd`'s
  `softmax_non_inner`, gets TILE_K=512 and 2.8x the bandwidth.
* `argmax_heur_tile_k` opens with `if M == 64 and K == 512`, a shape special
  case, and `benchmark/test_argmax.py` measures exactly that shape.

The rules below keep `_amd`'s TILE_K search, which grows the contiguous run per
workgroup until the grid no longer spans more than one wave, and replace the
part that was actually costing the most: num_warps. `_amd`'s reduction rule
returns 16 warps for any tile of 4096 elements or more, and its TILE_N keeps
tiles near 8192, so it returns 16 almost everywhere -- while the measured optima
on this part want 1 to 8. Aliasing these keys straight onto
`softmax_non_inner` is therefore not enough: it does repair mean, but it makes
argmax up to 3.9x slower than its best config, because `tl.max` with
`return_indices` carries an int64 index alongside every value and runs out of
registers long before a plain sum does.

Fitted against a tile sweep over 24 shapes on an idle gfx1201, held out against
the 10 shapes `benchmark/base.py:UnaryReductionBenchmark` measures. Median
distance from the best config found anywhere in the sweep is 1.05x for mean and
1.07x for argmax. See `run_results/tuning/ops/sweep_reduction_tiles.py`.
"""

import torch
import triton

# Tile area a workgroup handles per iteration, chosen from
# {1024, 2048, 4096, 8192, 16384}. The two keys want different answers because
# argmax has to keep an int64 index live next to every value it compares, so it
# runs out of registers on a tile that mean is still comfortable with: halving
# argmax's budget takes [8,8192,128] in fp32 from 0.74x torch to 0.97x, while
# the same halving costs mean 0.13x on [64,512,512].
_MEAN_TILE_BUDGET = 16384
_ARGMAX_TILE_BUDGET = 8192

# The sweep measured TILE_K up to 2048; do not extrapolate past it.
_MAX_TILE_K = 2048

# Elements per lane before another warp earns its keep.
_ELEMS_PER_LANE = 2048

_MAX_NUM_WARPS = 16


def _prev_power_of_2(n):
    return 1 << (n.bit_length() - 1) if n > 0 else 1


def _num_cus():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def reduction_heur_tile_k(args):
    """Widen the contiguous read until the grid stops covering the CUs twice over.

    grid_y is cdiv(K, TILE_K), so TILE_K trades blocks for coalescing. Start
    narrow and double while there is still more than one wave of workgroups to
    give up.
    """
    num_cus = _num_cus()
    upper_bound = min(args["K"], _MAX_TILE_K)
    tile_k = 1
    while tile_k <= upper_bound:
        num_blocks = args["M"] * triton.cdiv(args["K"], tile_k)
        if (num_blocks / num_cus > 1) and (tile_k * 2 <= upper_bound):
            tile_k *= 2
        else:
            break
    return tile_k


def _make_tile_n(tile_budget):
    """Spend what is left of the tile budget on the reduction axis.

    No floor beyond 1: rounding a short reduction axis up past N only buys a
    tile whose extra lanes are all masked off.
    """

    def reduction_heur_tile_n(args):
        per_row = max(1, tile_budget // args["TILE_K"])
        return max(1, min(triton.next_power_of_2(args["N"]), per_row))

    return reduction_heur_tile_n


def reduction_heur_one_tile_per_cta(args):
    return args["TILE_N"] >= args["N"]


def reduction_heur_num_warps(args):
    """One warp per _ELEMS_PER_LANE of tile.

    Deliberately far below the 16 that `_amd`'s softmax rule returns for tiles
    this size: these kernels are memory bound, and smaller workgroups let more
    of them sit resident per CU.
    """
    tile_size = args["TILE_N"] * args["TILE_K"]
    return max(1, min(_MAX_NUM_WARPS, _prev_power_of_2(tile_size // _ELEMS_PER_LANE)))


def _non_inner_reduction(tile_budget):
    # Order matters: triton.heuristics feeds each result into the args of the
    # next, so TILE_K has to be resolved before TILE_N, and both before
    # num_warps.
    return {
        "TILE_K": reduction_heur_tile_k,
        "TILE_N": _make_tile_n(tile_budget),
        "ONE_TILE_PER_CTA": reduction_heur_one_tile_per_cta,
        "num_warps": reduction_heur_num_warps,
    }


HEURISTICS_CONFIGS = {
    "mean_non_inner": _non_inner_reduction(_MEAN_TILE_BUDGET),
    "argmax_non_inner": _non_inner_reduction(_ARGMAX_TILE_BUDGET),
}
