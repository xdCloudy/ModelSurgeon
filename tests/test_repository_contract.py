"""Repository-level safety and governance contract tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    ".editorconfig",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/research.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml",
    ".github/workflows/ci.yml",
    ".gitignore",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "ROADMAP.md",
    "SECURITY.md",
    "pyproject.toml",
)

REQUIRED_IGNORE_PATTERNS = {
    ".env",
    ".venv/",
    "*.bin",
    "*.gguf",
    "*.safetensors",
    "artifacts/",
    "checkpoints/",
    "runs/",
    "work/",
}


def test_required_repository_files_are_present() -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]

    assert missing == [], f"required repository files are missing: {missing}"


def test_license_is_apache_2() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "Apache License" in license_text
    assert "Version 2.0" in license_text


def test_model_weights_secrets_and_experiment_outputs_are_ignored() -> None:
    configured_patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert configured_patterns >= REQUIRED_IGNORE_PATTERNS
