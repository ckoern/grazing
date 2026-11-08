"""Design parameters that are either fixed or varied within a range.

Parameters are immutable. Every modification returns a new object, which makes undo, caching
and handing parameters to background optimizations safe.

Parameter names are Python identifiers. Sets combined from several components carry dotted
names such as ``"cap.d"``; the dots are translated when converting to `lmfit.Parameters`.

Examples:
    >>> ps = ParameterSet([Parameter("d", 4.0, unit="nm"), Parameter("N", 30, integer=True)])
    >>> ps = ps.replace("N", vary=True, min=20, max=60)
    >>> ps["N"].vary, ps.free_names()
    (True, ('N',))
    >>> ps.value_dict()
    {'d': 4.0, 'N': 30.0}
"""

import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import lmfit

__all__ = [
    "Parameter",
    "ParameterSet",
    "ParameterValue",
    "Range",
    "expression_names",
    "free",
    "rename_in_expression",
    "resolve",
]

_LMFIT_SEP = "__"
_NAME_TOKEN = re.compile(r"(?<![\w.])[A-Za-z_][\w.]*")


def expression_names(expr: str) -> frozenset[str]:
    """Return the names an expression refers to.

    A name is a maximal run of identifier characters and dots that starts with a letter or
    underscore and does not directly follow a word character or a dot, so numbers such as
    ``1.5e3`` are not names. Function names such as ``sqrt`` are included.

    Args:
        expr: Expression text.

    Returns:
        The (possibly dotted) names.

    Examples:
        >>> sorted(expression_names("cap.d * 2 + sqrt(base) - 1.5e3"))
        ['base', 'cap.d', 'sqrt']
    """
    return frozenset(_NAME_TOKEN.findall(expr))


def rename_in_expression(expr: str, rename: Callable[[str], str]) -> str:
    """Replace every name in an expression.

    Args:
        expr: Expression text.
        rename: Maps a name (see `expression_names`) to its replacement. Names that should
            stay unchanged, such as function names, are mapped to themselves.

    Returns:
        The expression with all names replaced.

    Examples:
        >>> rename_in_expression("cap.d + capx.d", lambda n: n.replace("cap.", "top.", 1)
        ...                      if n.startswith("cap.") else n)
        'top.d + capx.d'
    """
    return _NAME_TOKEN.sub(lambda m: rename(m.group(0)), expr)


@dataclass(frozen=True)
class Parameter:
    """A named scalar design parameter.

    Args:
        name: Identifier, optionally dotted (``"cap.d"``).
        value: Current value. Must lie within [`min`, `max`].
        vary: Whether an optimizer may change the value. Fixed parameters keep `value`.
        min: Lower bound.
        max: Upper bound.
        expr: Expression in terms of other parameters of the same set (lmfit syntax, dotted
            names allowed). A parameter with an expression is neither fixed nor free; its
            value follows from the expression.
        integer: Whether only integer values are meaningful (e.g. the number of bilayers).
        unit: Unit of the value, e.g. ``"nm"``.
        description: Human-readable description.

    Raises:
        ValueError: If the name is invalid, the bounds are inconsistent, or the value is out of
            bounds or not integral for an integer parameter.
    """

    name: str
    value: float
    vary: bool = False
    min: float = -math.inf
    max: float = math.inf
    expr: str | None = None
    integer: bool = False
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not all(part.isidentifier() for part in self.name.split(".")) or not self.name:
            raise ValueError(f"invalid parameter name {self.name!r}")
        if _LMFIT_SEP in self.name:
            raise ValueError(f"parameter name {self.name!r} must not contain {_LMFIT_SEP!r}")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "min", float(self.min))
        object.__setattr__(self, "max", float(self.max))
        if math.isnan(self.value) or self.min > self.max:
            raise ValueError(f"parameter {self.name!r}: invalid value or bounds")
        if not self.min <= self.value <= self.max:
            raise ValueError(
                f"parameter {self.name!r}: value {self.value} outside [{self.min}, {self.max}]"
            )
        if self.integer and self.value != round(self.value):
            raise ValueError(f"parameter {self.name!r} must have an integer value")

    @property
    def is_free(self) -> bool:
        """Whether an optimizer varies this parameter (``vary`` and no expression)."""
        return self.vary and self.expr is None

    def evolve(self, **changes: Any) -> "Parameter":
        """Return a copy with some fields changed.

        If the bounds change but no new value is given, the value is moved into the new bounds
        (rounded up or down to an integer for integer parameters).

        Args:
            **changes: New values for any of the dataclass fields.

        Returns:
            The modified parameter.

        Raises:
            ValueError: If the result is invalid.
        """
        if "value" not in changes and ("min" in changes or "max" in changes):
            lo = float(changes.get("min", self.min))
            hi = float(changes.get("max", self.max))
            value = min(max(self.value, lo), hi)
            if self.integer and value != round(value):
                value = math.ceil(value) if value <= lo else math.floor(value)
            changes["value"] = value
        return replace(self, **changes)


@dataclass(frozen=True)
class Range:
    """Specification of a varied parameter, used when creating components.

    Args:
        value: Start value.
        min: Lower bound.
        max: Upper bound.
    """

    value: float
    min: float
    max: float


def free(value: float, min: float, max: float) -> Range:
    """Specify a parameter that is varied within [min, max].

    Args:
        value: Start value.
        min: Lower bound.
        max: Upper bound.

    Returns:
        The range specification.
    """
    return Range(value, min, max)


type ParameterValue = float | Range | str | Parameter
"""Ways to specify a parameter when creating a component.

A number fixes the parameter, a `Range` varies it, a string ties it to an expression, and a
`Parameter` replaces all settings except the name.
"""


def resolve(template: Parameter, spec: ParameterValue) -> Parameter:
    """Apply a user specification to a parameter template.

    Args:
        template: Parameter with name, unit, default bounds and description.
        spec: The user specification, see `ParameterValue`.

    Returns:
        The resulting parameter.

    Raises:
        ValueError: If the result is invalid.
        TypeError: If `spec` has an unsupported type.
    """
    if isinstance(spec, Parameter):
        return spec.evolve(name=template.name)
    if isinstance(spec, Range):
        return template.evolve(value=spec.value, vary=True, min=spec.min, max=spec.max, expr=None)
    if isinstance(spec, str):
        return template.evolve(expr=spec, vary=False)
    if isinstance(spec, int | float):
        lo, hi = min(template.min, spec), max(template.max, spec)
        return template.evolve(value=spec, vary=False, expr=None, min=lo, max=hi)
    raise TypeError(f"unsupported specification for {template.name!r}: {spec!r}")


class ParameterSet(Mapping[str, Parameter]):
    """An immutable, ordered collection of uniquely named parameters.

    Args:
        parameters: The parameters.

    Raises:
        ValueError: If names are not unique.
    """

    __slots__ = ("_params",)

    def __init__(self, parameters: Iterable[Parameter] = ()) -> None:
        params: dict[str, Parameter] = {}
        for p in parameters:
            if p.name in params:
                raise ValueError(f"duplicate parameter name {p.name!r}")
            params[p.name] = p
        self._params = params

    def __getitem__(self, name: str) -> Parameter:
        try:
            return self._params[name]
        except KeyError:
            raise KeyError(f"unknown parameter {name!r}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ParameterSet):
            return NotImplemented
        return tuple(self._params.values()) == tuple(other._params.values())

    def __hash__(self) -> int:
        return hash(tuple(self._params.values()))

    def __repr__(self) -> str:
        return f"ParameterSet({list(self._params.values())!r})"

    def value_dict(self) -> dict[str, float]:
        """Return the current value of every parameter.

        Expressions are not evaluated; see `evaluate` for that.
        """
        return {name: p.value for name, p in self._params.items()}

    def free_names(self) -> tuple[str, ...]:
        """Return the names of the parameters an optimizer varies."""
        return tuple(name for name, p in self._params.items() if p.is_free)

    def replace(self, name: str, **changes: Any) -> "ParameterSet":
        """Return a copy with one parameter modified.

        Args:
            name: Name of the parameter.
            **changes: New values for fields of the `Parameter`.

        Returns:
            The modified set.

        Raises:
            KeyError: If the parameter does not exist.
            ValueError: If the modified parameter is invalid.
        """
        new = self[name].evolve(**changes)
        return ParameterSet(new if p.name == name else p for p in self._params.values())

    def with_values(self, values: Mapping[str, float]) -> "ParameterSet":
        """Return a copy with new values for some parameters.

        Bounds are kept. Values of integer parameters are rounded.

        Args:
            values: New values by parameter name.

        Returns:
            The modified set.

        Raises:
            KeyError: If a name does not exist.
            ValueError: If a value is out of bounds.
        """
        unknown = set(values) - set(self._params)
        if unknown:
            raise KeyError(f"unknown parameters {sorted(unknown)}")
        return ParameterSet(
            p.evolve(value=round(values[n]) if p.integer else values[n]) if n in values else p
            for n, p in self._params.items()
        )

    def prefixed(self, prefix: str) -> "ParameterSet":
        """Return a copy with all names (and names in expressions) prefixed by ``prefix.``.

        Args:
            prefix: Namespace, a Python identifier.

        Returns:
            The renamed set.
        """
        names = set(self._params)

        def rename(word: str) -> str:
            return f"{prefix}.{word}" if word in names else word

        return ParameterSet(
            p.evolve(
                name=f"{prefix}.{p.name}",
                expr=rename_in_expression(p.expr, rename) if p.expr is not None else None,
            )
            for p in self._params.values()
        )

    def merge(self, *others: "ParameterSet") -> "ParameterSet":
        """Return the union of this set and others.

        Args:
            *others: Further sets, with names distinct from this set.

        Returns:
            The combined set, in order.

        Raises:
            ValueError: If names collide.
        """
        params = list(self._params.values())
        for other in others:
            params.extend(other.values())
        return ParameterSet(params)

    def to_lmfit(self, *, expression_bounds: bool = True) -> lmfit.Parameters:
        """Convert to `lmfit.Parameters`.

        Dots in names and expressions are replaced by ``"__"``. Expressions may refer to
        parameters defined later in the set.

        Args:
            expression_bounds: Apply the bounds of parameters with expressions. lmfit then clips
                expression results to these bounds.

        Returns:
            The lmfit parameters.
        """
        out = lmfit.Parameters()
        for p in self._params.values():
            bounded = expression_bounds or p.expr is None
            out.add(
                _to_lmfit_name(p.name),
                value=p.value,
                vary=p.is_free,
                min=p.min if bounded else -math.inf,
                max=p.max if bounded else math.inf,
            )
        for p in self._params.values():
            if p.expr is not None:
                out[_to_lmfit_name(p.name)].expr = rename_in_expression(p.expr, _to_lmfit_name)
        return out

    def update_from_lmfit(self, params: lmfit.Parameters) -> "ParameterSet":
        """Return a copy with values taken from `lmfit.Parameters`.

        Values of parameters with expressions are taken as well, so the result contains the
        evaluated expressions. Integer parameters are rounded.

        Args:
            params: Parameters as produced by `to_lmfit` (possibly optimized).

        Returns:
            The updated set.
        """
        values = {}
        for name, p in self._params.items():
            v = float(params[_to_lmfit_name(name)].value)
            values[name] = min(max(v, p.min), p.max)
        return self.with_values(values)

    def evaluate(self) -> dict[str, float]:
        """Return all values with expressions evaluated.

        Returns:
            Value of every parameter by name.

        Raises:
            ValueError: If an expression cannot be evaluated or its result lies outside the
                bounds of its parameter.
        """
        if all(p.expr is None for p in self._params.values()):
            return self.value_dict()
        try:
            lm = self.to_lmfit(expression_bounds=False)
            lm.update_constraints()
        except (NameError, SyntaxError, ValueError, TypeError) as exc:
            raise ValueError(f"cannot evaluate parameter expressions: {exc}") from None
        values = {name: float(lm[_to_lmfit_name(name)].value) for name in self._params}
        for name, p in self._params.items():
            if p.expr is not None and not p.min <= values[name] <= p.max:
                raise ValueError(
                    f"expression {p.expr!r} for {name!r} gives {values[name]:g}, "
                    f"outside [{p.min:g}, {p.max:g}]"
                )
        return values


def _to_lmfit_name(name: str) -> str:
    return name.replace(".", _LMFIT_SEP)
