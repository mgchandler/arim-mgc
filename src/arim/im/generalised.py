"""
Collection of functions related to generalised imaging and inverse imaging, towards extraction of the scattering matrix.
"""


import logging
import numba
import numpy as np
import os

from ..core import Frame, Time, View
from .. import geometry as g
from ..ray import RayGeometry
from ..model import beamspread_2d_for_path, reverse_beamspread_2d_for_path
from .tfm import FocalLaw
from . import das
from ..ut import hmc


logger = logging.getLogger(__name__)

use_parallel = os.environ.get("ARIM_USE_PARALLEL", not numba.core.config.IS_32BITS)


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


def inverse_for_generalised_image(
    generalised_image: GeneralisedResult,
    point_view: View,
    grid_view: View,
    pulse_width: float,
    frame: Frame,
    fillvalue: float = 0.0,
):
    """
    Compute the inverse image from the generalised image and the view.
    """
    tx, rx = hmc(point_view.tx_path.interfaces[0].points.numpoints)
    lookup_time = (
        point_view.tx_path.rays.times[tx]
        + point_view.rx_path.rays.times[rx]
        + np.arange(-pulse_width / 2, pulse_width / 2, frame.time.step).reshape(1, -1)
    ).transpose()
    time_on_grid = (
        grid_view.tx_path.rays.times[
            tx[:, None, None], generalised_image.tx[None, :, :]
        ]
        + grid_view.rx_path.rays.times[
            rx[:, None, None], generalised_image.rx[None, :, :]
        ]
    )
    result = np.zeros_like(lookup_time, dtype=generalised_image.res.dtype)
    tx_ray, rx_ray = RayGeometry.from_path(grid_view.tx_path), RayGeometry.from_path(
        grid_view.rx_path
    )
    amplitudes = (
        beamspread_2d_for_path(tx_ray)[
            tx[:, None, None], generalised_image.tx[None, :, :]
        ]
        * reverse_beamspread_2d_for_path(rx_ray)[
            rx[:, None, None], generalised_image.rx[None, :, :]
        ]
    )

    _das_linear(
        generalised_image.res,
        lookup_time,
        time_on_grid,
        amplitudes,
        fillvalue=fillvalue,
        result=result,
    )

    inv_const = (
        1
        / (2 * np.pi * point_view.tx_path.velocities[-1])
        * (generalised_image.grid.xvect[1] - generalised_image.grid.xvect[0]) ** 2
    )

    inverse_frame = Frame(
        np.zeros((tx.size, frame.time.samples.size), dtype=frame.timetraces.dtype),
        frame.time,
        tx,
        rx,
        frame.probe,
        frame.examination_object,
        frame.metadata,
    )
    time_window = np.abs(
        inverse_frame.time.samples.reshape(-1, 1)
        - lookup_time.min(axis=0).reshape(1, -1)
    ).argmin(axis=0)[..., None] + np.arange(lookup_time.shape[0])
    inverse_frame.timetraces[
        np.arange(tx.size)[:, None],
        time_window,
    ] = (
        result.transpose() * inv_const
    )
    inverse_frame = inverse_frame.expand_frame_assuming_reciprocity()

    return inverse_frame


# @numba.jit(nopython=True, nogil=True, parallel=use_parallel, fastmath=True)
def _das_linear(
    weighted_image,
    lookup_times,
    time,
    amplitudes,
    fillvalue,
    result,
):
    """
    Custom version of delay and sum from das, as it's too awkward to wrangle it into a
    Frame and FocalLaw. Note that in comparison to the matlab implementation, this
    function interpolates in time, rather than computing the vertical distance. I am not
    100% certain that this is equivalent, but the plots look pretty similar.
    """
    numfocals, numdepths = weighted_image.shape
    _, numscans = lookup_times.shape

    for scan in numba.prange(numscans):
        res_tmp = np.zeros_like(lookup_times[:, 0], dtype=weighted_image.dtype)

        for focal in range(numfocals):
            res_tmp += np.interp(
                lookup_times[:, scan],
                time[scan, focal, :],
                weighted_image[focal, :],
                left=fillvalue,
                right=fillvalue,
            ) * np.interp(
                lookup_times[:, scan],
                time[scan, focal, :],
                amplitudes[scan, focal, :],
                left=fillvalue,
                right=fillvalue,
            )

        result[:, scan] = res_tmp  # / numscans


def back_propagate():
    pass
