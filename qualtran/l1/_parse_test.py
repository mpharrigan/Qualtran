#  Copyright 2025 Google LLC
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
from qualtran import QUInt
from qualtran.l1._parse import BloqCode, parse_bloq_code


def test_parse_bloq_code():
    bloq_str = "qualtran.bloqs.my_bloq.MyBloq(arg1=5, arg2='hello', arg3=QUInt(5))"
    bloq_code = parse_bloq_code(bloq_str)
    assert bloq_code == BloqCode(
        package='qualtran.bloqs.my_bloq',
        bloq_class_name='MyBloq',
        args=[('arg1', 5), ('arg2', 'hello'), ('arg3', QUInt(5))],
    )


def test_parse_hubbard():
    s = "qualtran.bloqs.chemistry.hubbard_model.qubitization.PhaseEstimateHubbard(x_dim=2, y_dim=2, t=1.0, u=4.0, pe_bits=4)"
    bloq_code = parse_bloq_code(s)
    assert bloq_code == BloqCode(
        package='qualtran.bloqs.chemistry.hubbard_model.qubitization',
        bloq_class_name='PhaseEstimateHubbard',
        args=[('x_dim', 2), ('y_dim', 2), ('t', 1.0), ('u', 4.0), ('pe_bits', 4)],
    )
