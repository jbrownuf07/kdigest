"""GpuSession: one long-lived device context plus reusable GPU workspaces.

Every GPU entry point has a fixed per-call cost: a `DeviceContext` (~0.7 ms to
create), device buffers, and pinned host staging buffers (~0.2 ms each). A
session pays those once and amortizes them over every later call:

    var session = GpuSession()
    for batch in batches:
        digest.add_batch(batch, session)   # no per-call context / allocation

Workspaces grow geometrically to the largest batch seen and are then reused.
"""

from max.gpu.host import DeviceContext

from kdigest.gpu.grouped import GroupedWorkspace


struct GpuSession(Movable):
    var ctx: DeviceContext
    # Grouped buffers + the single-digest `CompressWorkspace` (`.batch`),
    # shared by `TDigest.add_batch` and the grouped oversized-key fallback.
    var ws: GroupedWorkspace

    def __init__(out self) raises:
        self.ctx = DeviceContext()
        self.ws = GroupedWorkspace(self.ctx)
