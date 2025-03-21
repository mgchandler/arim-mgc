"""
Collection of functions related to generalised imaging and inverse imaging, towards extraction of the scattering matrix.
"""


import logging
import numpy as np

from ..core import View
from .. import geometry as g
from .tfm import FocalLaw
from . import das
from ..ut import hmc


logger = logging.getLogger(__name__)


class GeneralisedResult:
    """
    Data container for generalised image result
    """

    __slots__ = ("res", "grid", "tx", "rx")

    def __init__(self, res, grid, tx, rx):
        assert res.shape == tx.shape
        assert tx.shape == rx.shape
        self.res = res
        self.grid = grid
        self.tx = tx
        self.rx = rx

    @classmethod
    def from_half_diagonal(cls, res, grid, tx, rx, numoffdiagonals):
        combinations = set(zip(tx.ravel(), rx.ravel()))
        grid_tx, grid_rx = generalised_indices(grid.numx, grid.numz, numoffdiagonals)
        combinations_grid1 = set(zip(grid_tx.ravel(), grid_rx.ravel()))
        combinations_grid2 = set(zip(grid_rx.ravel(), grid_tx.ravel()))
        if (tx.size == grid_tx.size) and (
            (combinations == combinations_grid1) or (combinations == combinations_grid2)
        ):
            tx = np.concatenate([tx, tx[grid.numx :, :]], axis=0)
            rx = np.concatenate([rx, rx[grid.numx :, :]], axis=0)
            res = np.concatenate([res, res[grid.numx :, :]], axis=0)
            return cls(res, grid, tx, rx)
        else:
            raise ValueError("Provided data is not half-diagonal.")


def generalised_indices(numx, numz, numoffdiagonals=None):
    """
    Return pairs of focal point indices for looking up in the grid-axis of path.rays.times (i.e. 1st axis).
    Default assumes all off-diagonals included, i.e. numoffdiagonals=numx
    """
    if numoffdiagonals is None:
        numoffdiagonals = numx

    # Index for repeated x in grid.
    # If timetraces were stored in 3D array (numelements, numx, numz) then these indices would be returned.
    x_tx = np.concatenate([range(numx - i) for i in range(numoffdiagonals + 1)])
    x_rx = np.concatenate([range(i, numx) for i in range(numoffdiagonals + 1)])

    # Actually, timetraces are stored in 2D array (numelements, numx * numz), so this has to be repeated for all z.
    # `grid.to_1d_points()` advances in z first, so preserve this order.
    tx = np.stack([np.arange(numz) + numz * x_tx[i] for i in range(x_tx.size)])
    rx = np.stack([np.arange(numz) + numz * x_rx[i] for i in range(x_rx.size)])

    return tx, rx


def generalised_image_for_view(
    frame,
    grid,
    view: View,
    amplitudes=None,
    mask=None,
    offdiagonallength=None,
    **kwargs_delay_and_sum,
):
    """
    Rearrange the FMC data into the generalised image for a provided view.

    Parameters
    ----------
    frame : Frame
    grid : Points
    velocity : float
    amplitudes : None or ndarray or TxRxAmplitudes
    mask : ndarray[bool]
        Mask which is applied to `grid.to_oriented_points()` when making views.
    kwargs_delay_and_sum : dict

    Returns
    -------
    tfm_res : TfmResult

    Notes
    -----
    No check is performed to ensure that the calculation is safe for incomplete frames (HMC for example).
    In this case, an :exc:`IncompleteFrameWarning` is emitted.

    """
    # do not use timetrace weights, it is likely to be ill-defined here
    if not frame.is_complete_assuming_reciprocity():
        logger.warning(
            "Possible erroneous usage of a noncomplete frame in TFM; "
            "use Frame.expand_frame_assuming_reciprocity()"
        )

    if offdiagonallength is None:
        numoffdiag = None
    else:
        numoffdiag = int(
            np.ceil(
                offdiagonallength
                * view.tx_path.velocities[-1]
                / frame.probe.frequency
                / grid.dx
            )
        )
    grid_tx, grid_rx = generalised_indices(grid.numx, grid.numz, numoffdiag)

    lookup_times_tx = view.tx_path.rays.times[:, grid_tx.ravel()].T
    lookup_times_rx = view.rx_path.rays.times[:, grid_rx.ravel()].T

    focal_law = FocalLaw(lookup_times_tx, lookup_times_rx, amplitudes)

    fw_const = -((frame.probe.locations[1][0] - frame.probe.locations[0][0]) ** 2) / (
        2 * np.pi * view.tx_path.velocities[-1]
    )
    res = das.delay_and_sum(frame, focal_law, **kwargs_delay_and_sum) * fw_const
    if type(grid) is g.Grid:
        res = res.reshape(grid_tx.shape)
    else:
        raise ValueError("Invalid grid type.")

    return GeneralisedResult.from_half_diagonal(res, grid, grid_tx, grid_rx, numoffdiag)


# def inverse_for_generalised_image(
#     generalised_image: GeneralisedResult,
#     point_view: View,
#     grid_view: View,
# ):
#     """
#     Compute the inverse image from the generalised image and the view.
#     """
#     tx, rx = hmc(point_view.tx_path.interfaces[0].size)
