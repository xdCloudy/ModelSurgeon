from collections.abc import Iterable

from modelsurgeon.graph import walk_named_modules


class Parameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class Module:
    def __init__(self, count: int) -> None:
        self._parameters = [Parameter(count)]

    def parameters(self, recurse: bool = True) -> list[Parameter]:
        assert recurse is False
        return self._parameters


class Model:
    def named_modules(self) -> Iterable[tuple[str, object]]:
        return [("", self), ("model.layers.0.mlp", Module(12))]


def test_walk_named_modules_builds_records() -> None:
    records = list(walk_named_modules(Model()))

    assert str(records[0].component_id) == "model"
    assert str(records[1].component_id) == "model.layers.0.mlp"
    assert records[1].parameter_count == 12

