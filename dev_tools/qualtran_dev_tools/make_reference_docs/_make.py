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
import re
import warnings
from pathlib import Path
from typing import cast, List

# Begin monkeypatch:
# the `Visitor` will use certain decorators to apply "labels" to the AST nodes, which
# causes properties to be parsed as Attributes instead of Functions. By removing these
# decorator-to-label mappings, they are kept as Function. We handle decorators for functions
# in this script how we want.
import _griffe.agents.visitor
import griffe
from griffe import (
    DocstringSection,
    DocstringSectionAdmonition,
    DocstringSectionKind,
    DocstringSectionParameters,
    DocstringSectionReturns,
    DocstringSectionText,
    GriffeLoader,
    Kind,
)

del _griffe.agents.visitor.builtin_decorators['property']
del _griffe.agents.visitor.stdlib_decorators['functools.cached_property']

PARSER = 'google'


def write_docstring_parts(f, parts: List[DocstringSection]):
    for part in parts:
        if part.kind is DocstringSectionKind.text:
            f.write(part.value.strip())
            f.write('\n\n')

        elif part.kind is DocstringSectionKind.parameters:
            part = cast(DocstringSectionParameters, part)
            f.write('### Args\n')
            for param in part.value:
                # f.write(f'{param.name=} {param.value=} {param.annotation=} {param.description=}')
                if param.annotation:
                    f.write(f'{param.name}: `{param.annotation}`\n')
                else:
                    f.write(f'{param.name}\n')
                f.write(f': {param.description}')  # myst syntax for definition lists in markdown
                f.write('\n\n')

        elif part.kind is DocstringSectionKind.returns:
            part = cast(DocstringSectionReturns, part)
            f.write('### Returns\n')

            # One, unnamed return value
            if len(part.value) == 1 and not part.value[0].name:
                f.write(part.value[0].description)
                f.write('\n')

            # Multiple return values and/or named return values
            else:
                for param in part.value:
                    # f.write(f'{param.name=} {param.value=} {param.annotation=} {param.description=}')
                    pname = param.name if param.name else 'ret'
                    f.write(f'{pname}\n')
                    f.write(
                        f': {param.description}'
                    )  # myst syntax for definition lists in markdown
                    f.write('\n\n')

        elif part.kind is DocstringSectionKind.admonition:
            part = cast(DocstringSectionAdmonition, part)
            part = part.value
            if part.kind == 'see-also':
                f.write('### See Also\n')
                f.write(part.description)
                f.write('\n\n')
            else:
                warnings.warn(f"Unknown admonition type {part.kind}")

        else:
            warnings.warn(f"Unknown docstring {part}")
    f.write('\n\n')


def format_parameter(p: griffe.Parameter) -> str:
    if p.annotation is not None and p.default is not None:
        s = f"{p.name}: {p.annotation} = {p.default}"
    elif p.annotation is not None and p.default is None:
        s = f"{p.name}: {p.annotation}"
    elif p.annotation is None and p.default is not None:
        s = f"{p.name} = {p.default}"
    else:
        s = f"{p.name}"

    if p.kind and p.kind is griffe.ParameterKind.positional_or_keyword:
        return s
    elif p.kind and p.kind is griffe.ParameterKind.positional_only:
        return s  # TODO?
    elif p.kind and p.kind is griffe.ParameterKind.keyword_only:
        return s  # TODO?
    elif p.kind and p.kind is griffe.ParameterKind.var_positional:
        return f'*{s}'
    elif p.kind and p.kind is griffe.ParameterKind.var_keyword:
        return f'**{s}'
    else:
        raise ValueError(p.kind)


def write_property_method_signature(f, obj, obj2: griffe.Function):
    assert obj2.returns, obj2
    obj_instance_name = camel_to_snake(obj.name)
    method_signature = f'{obj_instance_name}.{obj2.name} -> {obj2.returns}'
    f.write(f'```python\n{method_signature}\n```\n\n')


def camel_to_snake(name):
    # Turn an UpperCamelCaseName into lower_snake_case
    # .. we want to make it look like we're calling methods on instances of the class.
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return s1.lower()


def get_obj_instance_name(obj):
    # Get prototypical object instance names for a given class.

    # First, check if we have a bespoke name
    n = {'BloqBuilder': 'bb'}.get(obj.name, None)
    if n is not None:
        return n

    # Otherwise, use the class name as lower_snake_case.
    return camel_to_snake(obj.name)


def write_generic_method_signature(
    f, obj, obj2: griffe.Function, first_arg_name='self', caller_name=get_obj_instance_name
):
    # Strip `self` or `cls` argument.
    parameters = list(obj2.parameters)
    p0 = parameters[0]
    assert p0.name == first_arg_name, p0  # method
    parameters = parameters[1:]

    if len(parameters) > 1:
        # One line
        params = '\n' + ',\n'.join(f'  {format_parameter(p)}' for p in parameters) + '\n'
    else:
        # Multiline
        params = ', '.join(format_parameter(p) for p in parameters)

    # For methods, make it look like we're calling it on an instance.
    caller_name = caller_name(obj)

    # All together
    if obj2.returns is not None:
        method_signature = f'{caller_name}.{obj2.name}({params}) -> {obj2.returns}'
    else:
        method_signature = f'{caller_name}.{obj2.name}({params})'
    f.write(f'```python\n{method_signature}\n```\n\n')


def write_method_signature(f, obj, obj2: griffe.Function):
    if obj2.overloads:
        # todo .. handle overlaods
        warnings.warn(f'{obj.name}.{obj2.name} has {len(obj2.overloads)} overloads')

    # Dispatch to different writers based on decorators
    if obj2.decorators:

        # Filter out decorators that don't matter.
        decs = [d for d in obj2.decorators if str(d.value) not in ['abc.abstractmethod']]

        if len(decs) == 1:
            (d,) = decs

            # It's a property, write a special property signature.
            if str(d.value) == 'property' or str(d.value) == 'cached_property':
                return write_property_method_signature(f, obj, obj2)

            # It's a classmethod, write a special classmethod signature.
            elif str(d.value) == 'classmethod':
                return write_generic_method_signature(
                    f, obj, obj2, first_arg_name='cls', caller_name=lambda obj: obj.name
                )

            # Fallback: just skip it.
            else:
                # todo .. handle other decorators
                warnings.warn(f'{obj.name}.{obj2.name} has decorator {d.value!r}')
                return
        else:
            # todo .. handle more than one decorator.
            warnings.warn(f'{obj.name}.{obj2.name} has multiple decorators.')
            return

    # No decorators, write a normal method signature.
    return write_generic_method_signature(f, obj, obj2)


def _get_writers_for_split_docstring(obj: griffe.Object):
    """Extract the first, summary line of a docstring from an object.

    This returns two closures that take an `f` to write to. The first writes the first
    line and the second writes the rest. Both must be called.
    """

    if obj.docstring is None:

        def first_part(f):
            return

        def second_part(f):
            f.write('\n\n')

        return first_part, second_part

    dp0, *dparts = obj.docstring.parse(PARSER)
    if dp0.kind is DocstringSectionKind.text:
        first_line, *other_lines = re.split(r'\n{2,}', dp0.value, flags=re.MULTILINE)

        def first_part(f):
            f.write(first_line)
            f.write('\n\n')

        dp0 = DocstringSectionText('\n\n'.join(other_lines))
    else:

        def first_part(f):
            return

    def second_part(f):
        write_docstring_parts(f, [dp0] + dparts)

    return first_part, second_part


def write_major_class(f, obj: griffe.Class):
    # Title
    f.write(f"# {obj.name}\n")

    # Class docstring
    d0, drest = _get_writers_for_split_docstring(obj)
    d0(f)
    f.write('## Overview\n')  # Annoyingly have to include an <h2> before <h3> or sphinx freaks out
    drest(f)

    for name, obj2 in obj.members.items():
        if obj2.is_private:
            continue
        if obj2.name == '__init__':
            continue
        if obj2.is_special:
            # TODO: do we want these?
            continue
        if obj2.kind is Kind.ATTRIBUTE:
            # TODO: do we want these?
            continue

        # Member name
        f.write(f'## `{obj2.name}`\n')

        # First docstring line
        d0, drest = _get_writers_for_split_docstring(obj2)
        d0(f)

        # Optional: signature
        if obj2.kind is Kind.FUNCTION:
            write_method_signature(f, obj, cast(griffe.Function, obj2))

        # Rest of docstring
        drest(f)


def render_major_class(base_dir: Path, name: str, obj: griffe.Class | griffe.Alias):
    segments = obj.path.split('.')
    out_path = base_dir / '/'.join(segments[:-1]) / f'{segments[-1]}.md'
    print(f"Writing {name} to {out_path}")
    with out_path.open('w') as f:
        write_major_class(f, obj)


def make_reference_docs(p: Path):
    out_dir = p / 'docs/reference'

    loader = GriffeLoader()
    mod = loader.load("qualtran")
    assert mod.is_module
    assert mod.is_init_module

    for name, obj in mod.classes.items():
        # Render major_classes
        if name in ['Bloq', 'BloqBuilder', 'CompositeBloq', 'Signature', 'Register']:
            render_major_class(out_dir, name, obj)

        # ... todo .. render the rest of the module

        # ... todo .. recurse

    return
