import llm_observability


def test_version_is_exposed() -> None:
    assert llm_observability.__version__ == "0.1.0"
