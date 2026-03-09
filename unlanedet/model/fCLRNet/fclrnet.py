"""
FCLRNet — Vectorised variant of CLRNet.

Architecture (backbone / neck / aggregator / head) is identical to CLRNet.
The only difference is that FCLRHead replaces CLRHead, giving a vectorised
loss function and dynamic assignment.

Public interface is fully backward-compatible with CLRNet: configs and
trainers only need to swap the model class name to ``FCLRNet`` and the head
class name to ``FCLRHead``.
"""
from ..module import Detector


class FCLRNet(Detector):
    """
    Drop-in replacement for CLRNet.

    Set ``head`` to an ``FCLRHead`` instance in your config file.
    Everything else (backbone, neck, aggregator, forward logic) is
    inherited unchanged from ``Detector``.
    """

    def __init__(self, backbone=None, aggregator=None, neck=None, head=None):
        super().__init__(backbone, aggregator, neck, head)

    def forward(self, batch):
        return super().forward(batch)
