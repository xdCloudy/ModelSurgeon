FROM python:3.12-slim@sha256:e8a38829209aea82774882e5157e1f7c3e8c8393ff53a697b3580896e57f345c

ARG UV_VERSION=0.12.5
ARG EVALPLUS_REVISION=e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2

WORKDIR /opt/modelsurgeon-evalplus
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}"
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --extra evalplus --no-dev --no-install-project

ENV PATH="/opt/modelsurgeon-evalplus/.venv/bin:${PATH}"
ENV PYTHONDONTWRITEBYTECODE=1
LABEL org.modelsurgeon.evalplus.source="https://github.com/evalplus/evalplus"
LABEL org.modelsurgeon.evalplus.revision="${EVALPLUS_REVISION}"

CMD ["python"]
