# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="amd",
    device_name="cuda",
    device_query_cmd="rocm-smi",
)

"""
Mapping from the major version torch reports for an AMD GPU to the directory
holding that architecture's specialized configuration.

Only the major version is matched, so an entry covers a whole gfx family rather
than a single target. An architecture absent from the map falls back to the
vendor-wide configuration.

Example:
  gfx1200, gfx1201 (RDNA4) -> major 12 -> rdna4
"""

ARCH_MAP = {
    "12": "rdna4",
}

CUSTOMIZED_UNUSED_OPS = (
    "add",
    "cos",
    "cumsum",
)


__all__ = ["*"]
