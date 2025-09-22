#  Copyright 2024 Google LLC
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
import io
import re
import textwrap
from typing import Callable, List, Type

import qualtran as qlt


def main():
    bb = qlt.BloqBuilder()
    soqs = EqK_proc(bb, 8)
    cbloq = bb.finalize(**soqs)
    print(cbloq.signature)
    print(cbloq.debug_text())

    print("Getting imported stuff")
    import sys

    me = sys.modules[__name__]


def _get_source_parts(func):
    source = textwrap.dedent(inspect.getsource(func))

    # Usually, our bloq examples are constructed via functions annotated with the `@bloq_example`
    # annotation. If you want to find just the annotation: I develoepd this regex:
    # ma = re.match(r'@bloq_example.*?(?=^def)', source, flags=re.MULTILINE | re.DOTALL)
    #
    # Instead, we'll just strip anything until we find a `def(...) -> xxx:` line.
    # Regex explanation:
    #   - Non-greedy match any character until we get to def
    #   - Non-greedy match any character until we get to a return-type annotation
    #   - Non-greedy match any character until we get to the end of the type annotation
    ma_type_a = re.match(r'^.*?def .*?\) -> .*?:\n', source, flags=re.MULTILINE | re.DOTALL)
    ma_no_type_a = re.match(r'^.*?def .*?\):\n', source, flags=re.MULTILINE | re.DOTALL)

    if ma_type_a:
        ma = ma_type_a
    elif ma_no_type_a:
        ma = ma_no_type_a
    else:
        raise ValueError(f"{func} function source was not in the form we expected.")

    def_start, body_start = ma.span()
    assert def_start == 0, def_start
    return source[:body_start], source[body_start:]


def get_imports(mod):
    stuff = inspect.getmembers(mod)

    import_lines = set()
    cfg_lines = set()

    for name, obj in stuff:
        if inspect.isfunction(obj):
            if getattr(obj, 'is_mock_bloq', False):
                bloq_cls: Type[qlt.Bloq] = obj.bloq_cls
                *pkg, cls_name = bloq_cls._class_name_in_pkg_().split('.')
                pkg = '.'.join(pkg)
                import_lines.add(f'import {pkg}')
                cfg_lines.add(f'{name} = bb.cfg({bloq_cls._class_name_in_pkg_()})')
                # print(name, obj, obj.__name__, obj.bloq_cls)
    print('\n')

    for line in sorted(import_lines):
        print(line)
    print()
    for line in sorted(cfg_lines):
        print(line)

    return sorted(import_lines), sorted(cfg_lines)


def process_sugar(func: Callable):
    print(f"Processing {func}")
    mn = inspect.getmodule(func)
    print(f"Getting imports for {mn}")
    import_lines, cfg_lines = get_imports(mn)

    sig = inspect.signature(func)
    param_strs = ["bb: 'BloqBuilder'"]
    for param_name, param in sig.parameters.items():
        annot = param.annotation
        if param.annotation == inspect.Parameter.empty:
            raise ValueError(
                f"Un-annotated classical compile-time parameter type in {func}: {param_name}"
            )
        print(param_name, param, annot)
        param_strs.append(f'{param}')

    func_name = f'_{func.__name__}'
    f = io.StringIO()
    f.write(f"def {func_name}(")
    f.write(', '.join(param_strs))
    f.write("):\n")

    print(f"Inserting into source code")
    print('-' * 80)
    _, fbody = _get_source_parts(func)
    print()
    f.write('\n'.join(f'    {l}' for l in import_lines))
    f.write('\n\n')
    f.write('\n'.join(f'    {l}' for l in cfg_lines))
    f.write('\n\n')
    f.write('    ' + fbody.strip())

    print(f.getvalue())

    f = io.StringIO()
    cls_name = func.__name__
    func_args = ['bb'] + [f'{k}=self.{k}' for k in sig.parameters.keys()]
    func_args = ', '.join(func_args)

    f.write("@attrs.frozen\n")
    f.write(f"class {cls_name}(Bloq):\n")
    dd = ' ' * 4
    for pstr in param_strs[1:]:
        f.write(f'{dd}{pstr}\n')

    f.write(
        f"""
    @cached_property
    def signature(self) -> 'Signature':
        return self.decompose_bloq().signature

    def decompose_bloq(self) -> 'CompositeBloq':
        bb = qlt.BloqBuilder()
        soqs = {func_name}({func_args})
        return bb.finalize(**soqs)
"""
    )

    print('\n')
    print(f.getvalue())
