"""Task workspaces: a memory-backed, size-limited ``/work`` for one task (docs/specs/phase-1b.md).

The workspace is a tmpfs-backed docker volume (quota ``work_size_mb``, owned by the box user,
mode 0700). A tmpfs volume loses its contents as soon as no container has it mounted, so a
*holder* container (``sleep infinity`` under the full box hardening, with the volume mounted
read-only) keeps it mounted from :meth:`TaskWorkspace.open` to :meth:`TaskWorkspace.close`.
The workspace's data never touches the host's disk, and is gone once the workspace is closed.
"""

from __future__ import annotations

import contextlib
from types import TracebackType

from local_factory.config import SandboxConfig
from local_factory.sandbox.runtime import DockerRuntime, SandboxError
from local_factory.sandbox.spec import BOX_WORK, ContainerSpec, VolumeMount

__all__ = ["TaskWorkspace"]

HOLDER_ARGV: tuple[str, ...] = ("sleep", "infinity")


class TaskWorkspace:
    """A memory-backed, size-limited /work for one task. A tmpfs volume loses its contents when
    no container has it mounted, so a holder container keeps it mounted from open() to close()."""

    def __init__(self, runtime: DockerRuntime, config: SandboxConfig) -> None:
        self._runtime = runtime
        self._config = config
        self._volume: str | None = None
        self._holder: str | None = None

    def __repr__(self) -> str:
        return f"TaskWorkspace(volume={self._volume!r}, holder={self._holder!r})"

    @property
    def volume(self) -> str:
        """The workspace's volume name (only while the workspace is open)."""
        if self._volume is None:
            raise RuntimeError("the workspace is not open")
        return self._volume

    def open(self) -> TaskWorkspace:
        """Create the volume and start its holder. Idempotent."""
        if self._volume is not None:
            return self
        config = self._config
        self._volume = self._runtime.create_volume(
            size_mb=config.work_size_mb, uid=config.uid, gid=config.gid
        )
        try:
            holder = ContainerSpec.from_config(
                config, argv=HOLDER_ARGV, volumes=(self.mount(read_only=True),)
            )
            self._holder = self._runtime.start_detached(holder)
        except BaseException:
            volume, self._volume = self._volume, None
            # The original error matters more; a volume left behind is still labelled, so
            # DockerRuntime.cleanup() removes it.
            with contextlib.suppress(SandboxError):
                self._runtime.remove_volume(volume)
            raise
        return self

    def close(self) -> None:
        """Remove the holder, then the volume (its data is gone afterwards). Idempotent."""
        if self._holder is not None:
            self._runtime.remove_container(self._holder)
            self._holder = None
        if self._volume is not None:
            self._runtime.remove_volume(self._volume)
            self._volume = None

    def mount(self, *, read_only: bool = False) -> VolumeMount:
        """The workspace volume at ``/work``."""
        return VolumeMount(self.volume, BOX_WORK, read_only=read_only)

    def __enter__(self) -> TaskWorkspace:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
