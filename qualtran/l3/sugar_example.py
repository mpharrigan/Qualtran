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
import qualtran as qlt
from qualtran.l3 import In, Join, MultiControlX, process_sugar, Split, Xor, ZeroState


def EqK(n: int):
    x = In('x', qlt.QUInt(n))
    y = In('y', qlt.QUInt(n))

    x, y = Xor(x=x, y=y)
    ys = Split(reg=y)
    out = ZeroState()
    ys, out = MultiControlX(controls=ys, target=out)
    y = Join(qlt.QUInt(n), reg=ys)
    x, y = Xor(x=x, y=y)

    return {'x': x, 'y': y, 'out': out}


def main():
    # Should work without error, but doesn't do anything
    EqK(8)

    # Right now, prints python source code that turns it into a composite bloq / bloq.
    process_sugar(EqK)


if __name__ == '__main__':
    main()
