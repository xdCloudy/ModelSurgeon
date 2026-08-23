# Security Policy

## Supported versions

Until v1.0, only the latest `main` revision receives security fixes.

## Reporting

Please use GitHub private vulnerability reporting when available. Do not disclose exploitable model-loading, artifact-path, deserialization, or checkpoint-overwrite issues publicly before maintainers have assessed them.

## Security model

Model and dataset inputs are untrusted. Remote model code is disabled by default. ModelSurgeon must not overwrite source checkpoints by default, must use atomic output publication, and must constrain artifact paths to the configured run directory. Generated models can behave unexpectedly and require evaluation before use.

