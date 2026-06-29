import pytest

from src.utils.registry import Registry


def test_register_and_build():
    r = Registry("t")

    @r.register("a")
    def f(x=1):
        return x * 2

    assert r.build("a", x=3) == 6
    assert "a" in r
    assert r.names() == ["a"]


def test_default_name_is_object_name():
    r = Registry("t")

    @r.register()
    def myfunc():
        return 1

    assert "myfunc" in r


def test_duplicate_registration_raises():
    r = Registry("t")

    @r.register("a")
    def f():
        pass

    with pytest.raises(KeyError):

        @r.register("a")
        def g():
            pass


def test_unknown_name_raises():
    r = Registry("t")
    with pytest.raises(KeyError):
        r.build("missing")


def test_name_kwarg_does_not_collide():
    """Registry.build's `name` is positional-only, so a component may itself
    take a `name` keyword (as the Benchmark dataset does)."""
    r = Registry("t")

    @r.register("C")
    class C:
        def __init__(self, name, scale=1):
            self.name = name
            self.scale = scale

    obj = r.build("C", name="set5", scale=4)
    assert obj.name == "set5"
    assert obj.scale == 4
