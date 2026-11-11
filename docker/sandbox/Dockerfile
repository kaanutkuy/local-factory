# The box (docs/specs/phase-1b.md): every worker action, gate and export runs in this image.
# Orchestrator-owned. The factory's own code is NOT baked in: the running factory mounts its
# `local_factory` package read-only at /opt/local-factory, so the box always runs the same code
# as the host. The image provides only the runtime: Python 3.12, git, bash, pydantic.
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Pinned to uv.lock; the only third-party code the factory needs inside the box.
RUN pip install --no-cache-dir --no-compile \
      pydantic==2.14.3 pydantic-core==2.50.4 annotated-types==0.8.0 \
      typing-extensions==4.16.0 typing-inspection==0.4.5

# The box user. Never root, never the host user; the runtime passes --user explicitly anyway.
RUN groupadd --gid 10001 sandbox \
 && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /home/sandbox \
      --shell /usr/sbin/nologin sandbox

COPY lf-entry /usr/local/bin/lf-entry
RUN chmod 0755 /usr/local/bin/lf-entry

USER 10001:10001
WORKDIR /work
