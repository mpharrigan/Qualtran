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

"""A recursive-descent parser for bloq string representation."""
import importlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from qualtran import Bloq, QInt, QUInt


@dataclass
class Token:
    """A token from the input string."""

    type: str
    value: str
    line: int
    column: int


def tokenize(code: str) -> List[Token]:
    """Turn a string into a list of tokens."""
    token_specification = [
        ('NUMBER', r'\d+(\.\d*)?'),
        ('NAME', r'[A-Za-z_][A-Za-z_0-9]*'),
        ('STRING', r"'[^']*'|\"[^\"]*\""),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('EQUALS', r'='),
        ('COMMA', r','),
        ('DOT', r'\.'),
        ('NEWLINE', r'\n'),
        ('SKIP', r'[ \t]+'),
        ('MISMATCH', r'.'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    line_num = 1
    line_start = 0
    tokens = []
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        column = mo.start() - line_start
        if kind in ('NUMBER', 'NAME', 'LPAREN', 'RPAREN', 'EQUALS', 'COMMA', 'DOT'):
            tokens.append(Token(kind, value, line_num, column))
        elif kind == 'STRING':
            tokens.append(Token(kind, value[1:-1], line_num, column))  # remove quotes
        elif kind == 'NEWLINE':
            line_start = mo.end()
            line_num += 1
        elif kind == 'SKIP':
            pass
        elif kind == 'MISMATCH':
            raise RuntimeError(f'{value!r} unexpected on line {line_num}')
    return tokens


# AST Nodes
@dataclass
class LiteralNode:
    """AST node for a literal value."""

    value: Any


@dataclass
class BloqNode:
    """AST node for a bloq."""

    name: str
    args: List[Tuple[Optional[str], Any]]  # value can be LiteralNode or BloqNode


class Parser:
    """A recursive-descent parser for bloq strings."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        """Look at the next token without consuming it."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: Optional[str] = None) -> Token:
        """Consume the next token."""
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of input")
        if expected_type and token.type != expected_type:
            raise ValueError(f"Expected token type {expected_type} but got {token.type}")
        self.pos += 1
        return token

    def parse(self) -> BloqNode:
        """Parse the full token stream."""
        node = self.parse_bloq()
        if self.peek() is not None:
            raise ValueError("Extra tokens at the end of input")
        return node

    def parse_bloq(self) -> BloqNode:
        """Parse a bloq instantiation."""
        name = self.parse_identifier()
        args = []
        if self.peek() and self.peek().type == 'LPAREN':
            self.consume('LPAREN')
            if not (self.peek() and self.peek().type == 'RPAREN'):
                args = self.parse_arguments()
            self.consume('RPAREN')
        return BloqNode(name=name, args=args)

    def parse_identifier(self) -> str:
        """Parse a dot-separated identifier."""
        parts = [self.consume('NAME').value]
        while self.peek() and self.peek().type == 'DOT':
            self.consume('DOT')
            parts.append(self.consume('NAME').value)
        return '.'.join(parts)

    def parse_arguments(self) -> List[Tuple[Optional[str], Any]]:
        """Parse a list of arguments."""
        args = [self.parse_argument()]
        while self.peek() and self.peek().type == 'COMMA':
            self.consume('COMMA')
            args.append(self.parse_argument())
        return args

    def parse_argument(self) -> Tuple[Optional[str], Any]:
        """Parse a single argument (keyword or positional)."""
        if self.peek() and self.peek().type == 'NAME':
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'EQUALS':
                key = self.consume('NAME').value
                self.consume('EQUALS')
                value = self.parse_value()
                return key, value

        value = self.parse_value()
        return None, value

    def parse_value(self) -> Any:
        """Parse a value (literal or nested bloq)."""
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of input when parsing value")

        if token.type == 'NUMBER':
            self.consume('NUMBER')
            if '.' in token.value:
                return LiteralNode(value=float(token.value))
            return LiteralNode(value=int(token.value))
        if token.type == 'STRING':
            self.consume('STRING')
            return LiteralNode(value=token.value)
        if token.type == 'NAME':
            return self.parse_bloq()

        raise ValueError(f"Unexpected token {token} when parsing value")


@dataclass
class BloqCode:
    """A parsed bloq identifier."""

    package: str
    bloq_class_name: str
    args: List[Tuple[str, object]]

    def load(self) -> Bloq:
        """Load a bloq from a BloqCode object.

        Args:
            bloq_code: The parsed bloq code.

        Returns:
            An instantiated bloq.
        """
        module = importlib.import_module(self.package)
        bloq_cls = getattr(module, self.bloq_class_name)
        kwargs = {key: val for key, val in self.args}
        return bloq_cls(**kwargs)


EVAL_CONTEXT = {'QUInt': QUInt, 'QInt': QInt}


def _evaluate_node(node: Any, context: Dict[str, Any]) -> Any:
    """Recursively evaluate an AST node to a Python object."""
    if isinstance(node, LiteralNode):
        return node.value
    if isinstance(node, BloqNode):
        if node.name in context:
            cls = context[node.name]
            args = [_evaluate_node(arg[1], context) for arg in node.args if arg[0] is None]
            kwargs = {
                arg[0]: _evaluate_node(arg[1], context) for arg in node.args if arg[0] is not None
            }
            return cls(*args, **kwargs)
        raise ValueError(
            f"Unknown type '{node.name}' in arguments. Available types: {list(context.keys())}"
        )
    raise TypeError(f"Unknown AST node type: {type(node)}")


def parse_bloq_code(bloq_string: str) -> BloqCode:
    """Parse a string into a BloqCode object using a custom parser."""
    tokens = tokenize(bloq_string)
    parser = Parser(tokens)
    ast = parser.parse()

    if not isinstance(ast, BloqNode):
        raise TypeError("Expected a BloqNode at the top level")

    full_class_name = ast.name
    parts = full_class_name.split('.')
    package = '.'.join(parts[:-1])
    bloq_class_name = parts[-1]

    args = []
    for key, value_node in ast.args:
        if key is None:
            raise ValueError("Top-level bloq arguments must be keyword arguments.")
        value = _evaluate_node(value_node, EVAL_CONTEXT)
        args.append((key, value))

    return BloqCode(package=package, bloq_class_name=bloq_class_name, args=args)
