"""
BIDCell - biologically-informed deep learning for cell segmentation of subcelluar spatial transcriptomics data
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bidcell.BIDCellModel import BIDCellModel


__all__ = ["BIDCellModel"]


def __getattr__(name):
    """Load the high-level pipeline lazily so lightweight modules stay importable."""
    if name == "BIDCellModel":
        from bidcell.BIDCellModel import BIDCellModel

        return BIDCellModel
    raise AttributeError(name)
