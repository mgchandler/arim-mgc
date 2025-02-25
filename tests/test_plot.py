from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import arim
import arim.geometry as g
import arim.plot as aplt


@pytest.fixture
def plot_out_dir():
    d = Path("test_plots")
    d.mkdir(exist_ok=True)
    return d


def test_plot_oxz_many(plot_out_dir, show_plots):
    grid = arim.Grid(-5e-3, 5e-3, 0, 0, 0, 15e-3, 0.1e-3)
    k = 0.01e-3
    data = np.exp(-grid.x**2 / k - (grid.z - 5e-3) ** 2 / (2 * k))

    nrows = 2
    ncols = 3
    data_list = [data * (i + 1) for i in range(nrows * ncols)]
    title_list = [f"Plot {i + 1}" for i in range(nrows * ncols)]

    figsize = (12, 8)

    # Bare plot
    ax_list, im_list = aplt.plot_oxz_many(
        data_list, grid, nrows, ncols, figsize=figsize, axes_pad=0.2
    )
    plt.savefig(plot_out_dir / "test_plot_oxz_many_1.png")
    plt.close("all")

    # Add titles
    ax_list, im_list = aplt.plot_oxz_many(
        data_list,
        grid,
        nrows,
        ncols,
        title_list=title_list,
        suptitle="Many plots",
        figsize=figsize,
        y_suptitle=0.98,
        axes_pad=0.2,
    )
    plt.savefig(plot_out_dir / "test_plot_oxz_many_2.png")
    if show_plots:
        plt.show()
    else:
        plt.close("all")


def test_plot_oxz(plot_out_dir, show_plots):
    grid = arim.Grid(-5e-3, 5e-3, 0, 0, 0, 15e-3, 0.1e-3)
    k = 2 * np.pi / 10e-3
    data = (np.cos(grid.x * 2 * k) * np.sin(grid.z * k)) * (grid.z**2)

    # check it works without error
    ax, im = aplt.plot_oxz(data, grid)
    plt.close("all")

    ax, im = aplt.plot_oxz(
        data.reshape((grid.numx, grid.numz)),
        grid,
        scale="linear",
        title="some linear stuff",
    )
    plt.savefig(plot_out_dir / "test_plot_oxz_linear.png")
    if show_plots:
        plt.show()
    else:
        plt.close("all")

    ax, im = aplt.plot_oxz(
        data,
        grid,
        title="some db stuff",
        scale="db",
        clim=[-12, 0],
        savefig=True,
        filename=plot_out_dir / "test_plot_oxz_db.png",
    )
    if show_plots:
        plt.show()
    else:
        plt.close("all")


@pytest.mark.parametrize(
    "plot_interfaces_kwargs", [dict(), dict(show_orientations=True, show_last=True)]
)
def test_plot_interfaces(plot_out_dir, show_plots, plot_interfaces_kwargs):
    # setup interfaces
    numinterface = 200
    numinterface2 = 200

    xmin = -5e-3
    xmax = 60e-3
    z_backwall = 20e-3

    points, orientations = arim.geometry.points_1d_wall_z(
        0, 12e-3, z=0.0, numpoints=64, name="Probe"
    )
    rot = g.rotation_matrix_y(np.deg2rad(12))
    points = points.rotate(rot)
    points = points.translate((0, 0, -10e-3))
    orientations = orientations.rotate(rot)
    probe = arim.geometry.OrientedPoints(points, orientations)
    assert probe.orientations[0, 2, 0] > 0
    assert probe.orientations[0, 2, 2] > 0

    points, orientations = arim.geometry.points_1d_wall_z(
        xmin, xmax, z=0.0, numpoints=numinterface, name="Frontwall"
    )
    frontwall = arim.geometry.OrientedPoints(points, orientations)

    points, orientations = arim.geometry.points_1d_wall_z(
        xmin, xmax, z=z_backwall, numpoints=numinterface2, name="Backwall"
    )
    backwall = arim.geometry.OrientedPoints(points, orientations)

    grid_obj = arim.Grid(xmin, xmax, 0, 0, 0, z_backwall, 1e-3)
    grid = grid_obj.to_oriented_points()

    interfaces = [probe, frontwall, backwall, grid]
    # end setup interfaces

    aplt.plot_interfaces(interfaces, **plot_interfaces_kwargs)
    plt.savefig(plot_out_dir / "test_plot_interfaces.png")

    if show_plots:
        plt.show()
    else:
        plt.close("all")
