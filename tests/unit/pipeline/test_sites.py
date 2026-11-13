"""The public site constructors: `host_site()` and `box_site(sandbox)` hand out the sites the
slices use, so other pipelines (Gate 1) can run on the same ones."""

from __future__ import annotations

from typing import cast

from local_factory.config import SandboxConfig
from local_factory.gitops import BundleLimits
from local_factory.pipeline.host_slice import _HostSite, host_site
from local_factory.pipeline.sandboxed_slice import IMAGE_KEY, _BoxSite, box_site
from local_factory.sandbox import Sandbox

IMAGE_ID = "sha256:" + "e" * 64


class StubSandbox:
    def __init__(self) -> None:
        self.config = SandboxConfig(run_timeout_s=77.0, max_bundle_bytes=4321)
        self.checked = 0

    def check(self) -> str:
        self.checked += 1
        return IMAGE_ID

    def image_id(self) -> str:
        return IMAGE_ID


def test_host_site_is_the_plain_directory_site() -> None:
    site = host_site()
    assert isinstance(site, _HostSite)
    assert (site.run_limit_s, site.limits, site.image_digests()) == (None, BundleLimits(), {})
    site.check()
    assert host_site() is not site  # no shared state between users of the site


def test_box_site_is_the_sandbox_s_site() -> None:
    stub = StubSandbox()
    site = box_site(cast("Sandbox", stub))
    assert isinstance(site, _BoxSite)
    assert site.run_limit_s == 77.0
    assert site.limits == BundleLimits(max_bundle_bytes=4321)
    assert site.image_digests() == {IMAGE_KEY: IMAGE_ID}
    site.check()
    assert stub.checked == 1
