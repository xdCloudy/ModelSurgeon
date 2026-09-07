"""Contract tests for the v1.0 end-to-end user guides."""

from pathlib import Path

from typer.testing import CliRunner

from modelsurgeon.cli.app import app

ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    ROOT / "docs" / "user-guides" / "huggingface.md",
    ROOT / "docs" / "user-guides" / "native-gguf.md",
)


def test_guides_exist_and_preserve_the_safety_contract() -> None:
    for guide in GUIDES:
        text = guide.read_text(encoding="utf-8")
        assert "immutable" in text.lower()
        assert "never" in text.lower() and "source" in text.lower()
        assert "fail" in text.lower() and "closed" in text.lower()
        assert "reproduce" in text.lower()


def test_huggingface_guide_covers_release_commands_and_atomic_output() -> None:
    text = GUIDES[0].read_text(encoding="utf-8")

    assert "uv sync --extra dev --extra hf --locked" in text
    assert "modelsurgeon inspect" in text
    assert "first-surgeon-hf-proof" in text
    assert "modelsurgeon calibrate" in text
    assert "write_safetensors_checkpoint_atomic" in text
    assert "destination" in text and "source" in text


def test_native_gguf_guide_covers_release_commands_and_resume() -> None:
    text = GUIDES[1].read_text(encoding="utf-8")

    assert "uv sync --extra dev --locked" in text
    assert "Q4_K_M" in text
    assert "plan_native_gguf_model_mlp_channel_removal" in text
    assert "execute_native_gguf_mlp_channel_removal" in text
    assert "discard_resumable_gguf" in text
    assert "llama-cli" in text


def test_documented_cli_help_contracts_pass_on_this_checkout() -> None:
    runner = CliRunner()
    for command in ("inspect", "first-surgeon-hf-proof", "calibrate", "reproduce"):
        result = runner.invoke(app, [command, "--help"], color=False)
        assert result.exit_code == 0, result.output
        assert "Usage:" in result.output
        assert "\x1b" not in result.output
