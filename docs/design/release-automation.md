# Release automation

The release workflow is deliberately split into build, attestation, and publish stages.

## Release trigger and version identity

A maintainer creates and publishes a GitHub Release for a tag such as v1.0.0. The workflow checks out that tag and requires it to match the version in pyproject.toml. A release cannot silently package a different commit or version. Version bumps and the corresponding versioned CHANGELOG.md section are explicit repository changes.

## Build and clean-install gate

The build job creates exactly one sdist and one wheel with uv build. verify_release_artifacts.py checks archive safety, package names, versions, Python requirements, package contents, and the release tag. audit_v1_release.py checks the versioned release contract, release notes, and content-addressed reference manifests. generate_release_notes.py renders the matching versioned changelog section without changing CHANGELOG.md.

The wheel is installed into a new Python 3.12 environment before the workflow can continue. The smoke imports the package and CLI app; dependency resolution remains the responsibility of the package index and the wheel metadata.

## Artifact attestation

The attest job receives only the verified archives and emits GitHub build-provenance attestations. The repository does not store signing keys or long-lived package credentials. PyPI trusted publishing supplies the short-lived OIDC identity after the publish job's environment gate.

## Approval-gated publication

The publish job requires the repository environment named pypi. Configure that environment with required reviewers and a PyPI trusted publisher for this repository and workflow. The job publishes the sdist and wheel only after build, clean installation, attestation, and the environment approval succeed. It then attaches the archives and generated notes to the GitHub Release.

If a release fails, do not overwrite an existing PyPI version. Fix the repository and publish a new version/tag after rerunning the complete gate. An attestation or GitHub Release attachment is not evidence that a PyPI upload succeeded; retain each job's result separately.
