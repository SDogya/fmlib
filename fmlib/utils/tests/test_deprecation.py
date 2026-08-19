import pytest

from fmlib.utils.deprecation import deprecation_warning


def test_deprecation_class() -> None:
    @deprecation_warning()
    class DeprecatedClass:
        def __init__(self, *args, **kwargs):
            self.state = (args, kwargs)

    with pytest.warns(DeprecationWarning, match="DeprecatedClass"):
        result = DeprecatedClass(1, 2, 3, foo="bar")
    assert result.state == ((1, 2, 3), {"foo": "bar"})


def test_deprecation_function() -> None:
    @deprecation_warning()
    def deprecated_function(*args, **kwargs):
        return (args, kwargs)

    with pytest.warns(DeprecationWarning, match="function"):
        result = deprecated_function(1, 2, 3, foo="bar")
    assert result == ((1, 2, 3), {"foo": "bar"})


@pytest.mark.parametrize("text", ["foo", "bar", "deadbeaf"])
def test_deprecation_custom_message(text: str) -> None:
    @deprecation_warning(text)
    class DeprecatedClass:
        def __init__(self, *args, **kwargs):
            self.state = (args, kwargs)

    with pytest.warns(DeprecationWarning, match=text):
        result = DeprecatedClass(1, 2, 3, foo="bar")
    assert result.state == ((1, 2, 3), {"foo": "bar"})
