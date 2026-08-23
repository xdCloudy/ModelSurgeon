from modelsurgeon.config import Settings


def test_safe_defaults() -> None:
    settings = Settings()

    assert settings.safety.allow_overwrite is False
    assert settings.safety.trust_remote_code is False
    assert settings.hardware.cpu_offload is True

