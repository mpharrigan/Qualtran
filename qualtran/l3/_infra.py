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
from typing import Any, Dict, Optional, Protocol, Sequence, Set, Union

import attrs
import numpy as np

from qualtran import Bloq, BloqBuilder, BloqError, CompositeBloq, Register, Signature
from qualtran._infra.composite_bloq import QVarT
from qualtran._infra.quantum_graph import _QVar


class _TracingBloqFuncT(Protocol):
    def __call__(self, bb: 'BloqBuilder', *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """the structure of the function.

        During normal operation
         - a `bb: BloqBuilder` will be passed positionally as the first argument.
         - all other quantum and classical args and kwargs will be bound according to the Python
           rules and passed to the function by keyword. There are no additional positional-only
           arguments allowed (other than optionally `bb`).

        During `make`
         - a `bb: BloqBuilder` will be passed positionally as the first argument.
         - positional arguments will the be passed
         - any provided keyword arguments will be merged with a dictionary of initial quantum
           variables constructed by this method, and passed by keyword. An error is raised
           if a keyworkd argument is provided that interferes with this.

         - During normal operation, all quantum and classical arguments will be
         - Quantum arguments will always be passed by keyword, so you should probably put them last.
        """


@attrs.mutable
class _BloqifyPrepResult:
    cbloq: 'CompositeBloq'
    in_qargnames: Set[str]
    out_qargnames: Set[str]
    explicit_bb: Optional['BloqBuilder'] = None
    found_bb: Optional['BloqBuilder'] = None

    @property
    def bb(self) -> 'BloqBuilder':
        if self.explicit_bb is not None:
            return self.explicit_bb
        if self.found_bb is not None:
            return self.found_bb

        raise ValueError(
            "Could not find a valid bloq builder object to qcall this function. "
            "Please add an explicit `bb: BloqBuilder` argument to your `@bloqify` function."
        )


_BB_PLACEHOLDER = object()


class _TracingBloqIntermediate:
    def __init__(self, func: _TracingBloqFuncT):
        # TODO: better error when func doesn't take bb: BloqBuilder as first argument.
        # TODO: TypeError: cselect1() got multiple values for argument 'ctrl'
        self.func: _TracingBloqFuncT = func

    @property
    def name(self) -> str:
        return self.func.__name__

    @property
    def pkg(self) -> str:
        return self.func.__module__

    def _bound_kvs(self, *args, **kwargs):
        yield from inspect.signature(self.func).bind(
            _BB_PLACEHOLDER, *args, **kwargs
        ).arguments.items()

    def _is_qvar_array(self, v):
        if not isinstance(v, (Sequence, np.ndarray)):
            return False

        try:
            x = np.asarray(v).reshape(-1).item(0)
            return isinstance(x, _QVar)
        except (ValueError, IndexError):
            return False

    def _prep_qstackframe(self, kv_iter):
        bb = BloqBuilder(bloq_key=self.name, add_registers_allowed=True)
        qkwargs = {}
        ckwargs = {}

        for k, v in kv_iter:
            # TODO: handle shaped
            if v is _BB_PLACEHOLDER:
                continue
            elif isinstance(v, _QVar):
                qkwargs[k] = bb.in_register(name=k, dtype=v.dtype)
            elif self._is_qvar_array(v):
                v = np.asarray(v)
                qkwargs[k] = bb.in_register(
                    name=k, dtype=v.reshape(-1).item(0).dtype, shape=v.shape
                )
            else:
                ckwargs[k] = v

        out_qvars = self.func(bb, **ckwargs, **qkwargs)
        if not isinstance(out_qvars, dict):
            raise ValueError(
                f"{self.name} is expected to return a dictionary mapping "
                f"output register name to output quantum variable."
            )
        cbloq = bb.finalize(**out_qvars)
        return _BloqifyPrepResult(
            cbloq=cbloq, in_qargnames=set(qkwargs.keys()), out_qargnames=set(out_qvars.keys())
        )

    def make(self, signature: 'Signature', *classical_args, **classical_kwargs):
        bb, soqs = BloqBuilder.from_signature(
            signature, bloq_key=self.name, add_registers_allowed=True
        )

        dupes = set(classical_kwargs.keys()) & set(soqs.keys())
        if dupes:
            raise ValueError(
                f"`make` called with keyword arguments that shadow quantum "
                "register names: {dupes}. Please do not provide quantum variables "
                "when calling `make`."
            )

        kwargs = classical_kwargs | soqs
        soqs = self.func(bb, *classical_args, **kwargs)
        return bb.finalize(**soqs)

    def make_bloq_class(self, signature, **fields):
        cls_name = self.name
        ## TODO: infer fields from self.func

        def _signature(s2):
            return signature

        def _decompose_bloq(s2):
            return self.make(signature, **{k: getattr(s2, k) for k in fields})

        class_body = {
            'signature': property(_signature),
            'decompose_bloq': _decompose_bloq,
            '__doc__': f"A bloqified bloq class '{self.name}'.",
        }

        return attrs.make_class(
            name=cls_name, attrs=fields, bases=(Bloq,), class_body=class_body, frozen=True
        )

    def __call__(self, bb: 'BloqBuilder', /, *args, **kwargs):
        # try:
        f = self._prep_qstackframe(self._bound_kvs(*args, **kwargs))
        # except BloqError as be:
        #     raise BloqError(*be.args) from None

        return bb.add(
            f.cbloq, **{k: v for k, v in self._bound_kvs(*args, **kwargs) if k in f.in_qargnames}
        )

    def adjoint(self, bb: 'BloqBuilder', /, *args, **kwargs):
        if args:
            raise ValueError(
                "`.adjoint` must be called with keyword arguments: the outputs are now inputs, and we can't inspect outputs."
            )
        # TODO: fundamentally broken
        f = self._prep_qstackframe(kwargs.items())
        return bb.add(
            f.cbloq.adjoint(), **{k: v for k, v in kwargs.items() if k in f.out_qargnames}
        )

    def inline(self, bb: 'BloqBuilder', /, *args, **kwargs):
        ret_dict = self.func(bb, *args, **kwargs)
        return tuple(ret_dict.values())


def bloqify(func: _TracingBloqFuncT) -> _TracingBloqIntermediate:
    return _TracingBloqIntermediate(func)
