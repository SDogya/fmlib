from dataclasses import dataclass, field
from typing import Any, Dict, List, Self


@dataclass
class TypedBuilder:
    type: type

    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self: Self, *args, **kwargs) -> Any:
        all_args: List[Any] = [*args, *self.args]
        all_kwargs: Dict[str, Any] = {**kwargs, **self.kwargs}
        return (self.type)(*all_args, **all_kwargs)
