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

import abc
from typing import Iterable, Optional, Sequence, Tuple, TYPE_CHECKING

import attrs

from qualtran._infra.bloq import Bloq
from qualtran._infra.controlled import CtrlSpec
from qualtran._infra.registers import Register

if TYPE_CHECKING:
    from qualtran import AddControlledT, BloqBuilder, SoquetT


def get_single_reg_ctrl_system(
    ctrl_bloq: 'Bloq', ctrl_reg_name: str
) -> Tuple['Bloq', 'AddControlledT']:
    def adder(
        bb: 'BloqBuilder', ctrl_soqs: Sequence['SoquetT'], in_soqs: dict[str, 'SoquetT']
    ) -> tuple[Iterable['SoquetT'], Iterable['SoquetT']]:
        (ctrl_soq,) = ctrl_soqs
        soqs = {ctrl_reg_name: ctrl_soq} | in_soqs
        soqs = bb.add_d(ctrl_bloq, **soqs)
        ctrl_soqs = [soqs.pop(ctrl_reg_name)]
        return ctrl_soqs, soqs.values()

    return ctrl_bloq, adder


class SpecializedSingleQubitControlledExtension(Bloq):
    """A bloq with a specialized single-qubit controlled version.

    Bloq authors can inherit from this class to provide a specialized version of the
    construction that handles 0 or 1 bits of control.

    The implementing class must have an attribute `control_val` which configures the single-bit
    control. When `control_val` is not None, the `control_registers` property should return a
    single named qubit register, and otherwise return an empty tuple.

    Example usage:

        @attrs.frozen
        class MyGate(SpecializedSingleQubitControlledExtension):
            control_val: Optional[int] = None

            @property
            def control_registers() -> Tuple[Register, ...]:
                return () if self.control_val is None else (Register('control', QBit()),)

    Alternatively, bloq authors can use the static methods on this class to manually configure
    a control system without using inheritance.
    """

    control_val: Optional[int]

    @staticmethod
    def ctrl_system_helper(
        ctrl_bloq: 'Bloq', ctrl_reg_name: str
    ) -> Tuple['Bloq', 'AddControlledT']:
        """A static method for helping explicitly write your own `get_ctrl_system`.

        Bloq authors can set up a controlled version of the bloq with a known control register
        name, and use this function to easily return the callable required by `get_ctrl_system`.

        Args:
            ctrl_bloq: The controlled version of the bloq
            ctrl_reg_name: The name of the new register that takes a control soquet.

        Returns:
            ctrl_bloq: The control bloq, per the `Bloq.get_ctrl_system` interface.
            add_controlled: A function that adds the controlled version of the bloq to
                a composite bloq that is being built, per the `Bloq.get_ctrl_system` interface.
        """
        return get_single_reg_ctrl_system(ctrl_bloq, ctrl_reg_name)

    @property
    @abc.abstractmethod
    def control_registers(self) -> Tuple[Register, ...]: ...

    def get_single_qubit_controlled_bloq(
        self, control_val: int
    ) -> 'SpecializedSingleQubitControlledExtension':
        """Override this to provide a custom controlled bloq"""
        return attrs.evolve(self, control_val=control_val)  # type: ignore[misc]

    def get_ctrl_system(self, ctrl_spec: 'CtrlSpec') -> Tuple['Bloq', 'AddControlledT']:
        if self.control_val is None and ctrl_spec.shapes in [((),), ((1,),)]:
            control_val = int(ctrl_spec.cvs[0].item())
            ctrl_bloq = self.get_single_qubit_controlled_bloq(control_val)

            if not hasattr(ctrl_bloq, 'control_registers'):
                raise TypeError(f"{ctrl_bloq} should have attribute `control_registers`")

            (ctrl_reg,) = ctrl_bloq.control_registers
            return SpecializedSingleQubitControlledExtension.ctrl_system_helper(
                ctrl_bloq=ctrl_bloq, ctrl_reg_name=ctrl_reg.name
            )

        return super().get_ctrl_system(ctrl_spec)
