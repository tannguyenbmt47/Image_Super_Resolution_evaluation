from src.utils import load_config
from src.utils.config import Config


def test_attribute_access():
    c = Config({"a": 1, "b": {"c": 2}})
    assert c.a == 1
    assert c.b.c == 2


def test_get_with_default():
    c = Config({"a": 1})
    assert c.get("a") == 1
    assert c.get("missing", 5) == 5


def test_list_of_dicts_is_wrapped():
    # mirrors real configs: a list of dataset specs accessed by attribute
    c = Config({"datasets": [{"name": "Set5"}, {"name": "Set14"}]})
    assert c.datasets[0].name == "Set5"
    assert c.datasets[1].name == "Set14"


def test_dict_method_keys_need_indexing():
    # documents the dict-subclass limitation: `items` resolves to the method,
    # but the value is still reachable by indexing.
    c = Config({"items": [1, 2]})
    assert c["items"] == [1, 2]
    assert callable(c.items)


def test_load_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  name: SR3\n  args:\n    scale: 4\n")
    cfg = load_config(p)
    assert cfg.model.name == "SR3"
    assert cfg.model.args.scale == 4
