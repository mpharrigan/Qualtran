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

import argparse
from typing import List

from qualtran_dev_tools.bloq_finder import get_bloq_classes, get_bloq_examples

import qualtran.testing as qlt_testing
from qualtran import BloqExample


def get_bloq_examples_for_bloq_class(bloq_cls_name: str) -> List[BloqExample]:
    bclass = {bc.__name__: bc for bc in get_bloq_classes()}[bloq_cls_name]
    bexamples = [be for be in get_bloq_examples() if be.bloq_cls == bclass]
    return bexamples


def score_bloq_example(be: BloqExample) -> None:
    print(be.name)
    res, err = qlt_testing.check_bloq_example_make(be)
    print('qlt_testing.check_bloq_example_make:             ', res.name, err)
    res, err = qlt_testing.check_bloq_example_decompose(be)
    print('qlt_testing.check_bloq_example_decompose:        ', res.name, err)
    res, err = qlt_testing.check_equivalent_bloq_example_counts(be)
    print('qlt_testing.check_equivalent_bloq_example_counts:', res.name, err)


def main(bloq_cls_name: str) -> None:
    print(f"Finding examples for {bloq_cls_name}...", flush=True)
    bexamples = get_bloq_examples_for_bloq_class(bloq_cls_name)
    print(f"Checking {len(bexamples)} instantiations...", flush=True)
    print()
    for be in bexamples:
        score_bloq_example(be)
        print()


def parse_args() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('bloq_cls_name')
    args = parser.parse_args()
    main(bloq_cls_name=args.bloq_cls_name)


if __name__ == '__main__':
    parse_args()
