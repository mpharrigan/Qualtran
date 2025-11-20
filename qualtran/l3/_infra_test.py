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
import inspect


def func1(x, k, n=1):
    pass


def call_func1_with_kwargs(*args, **kwargs):
    stuff = inspect.signature(func1).bind(*args, **kwargs).arguments
    return dict(stuff)


def test_inspect():
    a = call_func1_with_kwargs('xv', 'kv', n=2)
    assert a == {'x': 'xv', 'k': 'kv', 'n': 2}

    # Note: we could use `.apply_defaults` if we wanted the default values
    a = call_func1_with_kwargs('xv', 'kv')
    assert a == {'x': 'xv', 'k': 'kv'}

    a = call_func1_with_kwargs('xv', n=2, k=1)
    assert a == {'x': 'xv', 'n': 2, 'k': 1}
