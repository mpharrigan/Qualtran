#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import sympy

import qualtran.testing as qlt_testing
from qualtran.bloqs.factoring.ecc import ECPoint
from qualtran.bloqs.factoring.ecc.ec_add_r import _ec_add_r, _ec_add_r_small, _ec_window_add, ECAddR


def test_ec_add_r(bloq_autotester):
    bloq_autotester(_ec_add_r)


def test_ec_add_r_small(bloq_autotester):
    bloq_autotester(_ec_add_r_small)


def test_ec_window_add(bloq_autotester):
    bloq_autotester(_ec_window_add)


def test_ec_add_r_classical():
    n = 5  # fits our mod = 17
    P = ECPoint(15, 13, mod=17, curve_a=0)
    Q = ECPoint(2, 4, mod=17, curve_a=0)
    p1 = ECPoint(5, 8, mod=17, curve_a=0)
    p2 = ECPoint(10, 15, mod=17, curve_a=0)
    ec_add_r_small = ECAddR(n=n, R=p1)

    # x1, y1, x2, y2, p = sympy.symbols('x1 y1 x2 y2 p', integer=True, positive=True)
    # ec_add_r_small = ECAddR(n=n, R=ECPoint(x1, y1, p))
    # ref_ctrl, ref_x, ref_y = ec_add_r_small.call_classically(ctrl=1, x=x2, y=y2)
    # cbloq = ec_add_r_small.decompose_bloq()
    # decomp_ctrl, decomp_x, decomp_y = cbloq.call_classically(ctrl=1, x=x2, y=y2)
    # assert ref_ctrl == decomp_ctrl
    # assert (ref_x, ref_y) == (decomp_x, decomp_y)

    ref_ctrl, ref_x, ref_y = ec_add_r_small.call_classically(ctrl=1, x=p2.x, y=p2.y)
    cbloq = ec_add_r_small.decompose_bloq()
    qlt_testing.assert_valid_cbloq(cbloq)
    decomp_ctrl, decomp_x, decomp_y = cbloq.call_classically(ctrl=1, x=p2.x, y=p2.y)
    assert ref_ctrl == decomp_ctrl
    assert (ref_x, ref_y) == (decomp_x, decomp_y)
