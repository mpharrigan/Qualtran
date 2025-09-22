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

# Quantum program 2
# Negate a quantum integer
#
# Registers:
#   x: an 8-bit signed quantum integer


import qualtran.l3 as qlt


def negate(n: int):
    x = qlt.In('x', qlt.QInt(n))
    x = ~x
    x += 1
    return {'x': x}


def main():
    qlt.process_sugar(negate)


if __name__ == '__main__':
    main()
