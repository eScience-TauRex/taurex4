"""Memory management utilities.

Provides helpers for releasing freed memory back to the operating system.
Python and numpy use internal memory pools that hold onto freed pages,
which causes RSS to grow even when actual allocations are stable.

The implementation now lives in :mod:`taurex.util.util`, which is shipped by
every install (including older ones), so that importing these helpers cannot
fail because a newly added module was left out of an environment. This module
is kept only as a backwards-compatible import path.
"""

from taurex.util.util import memory_usage_mb
from taurex.util.util import trim_memory


__all__ = ["trim_memory", "memory_usage_mb"]
