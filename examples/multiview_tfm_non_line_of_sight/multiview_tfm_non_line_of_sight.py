#!/usr/bin/env python3
"""
Multi-view TFM for a contact inspection

If the configuration entry "reference_rect" is defined, the dB scale for each
view will be adjusted to the maximum intensity in the area.

A backwall must be defined for imaging with skip paths.
A frontwall and a backwall must be defined for imaging with double-skip paths.


Input
-----
conf.yaml
    Configuration file

Output
------
TFM images

"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import arim
import arim.geometry as g
import arim.im
import arim.io
import arim.models.block_in_contact as bic
import arim.plot as aplt
import arim.ray
import arim.signal

# %% Load configuration

conf = arim.io.load_conf(".")
save = False
aplt.conf["savefig"] = save
result_dir = conf["result_dir"]
imaging_walls = conf["contiguous_geometry"]["imaging_walls"]

# %% Load frame
frame = arim.io.frame_from_conf(conf)
frame = frame.apply_filter(
    arim.signal.Hanning(
        frame.numsamples, frame.probe.frequency, frame.probe.frequency, frame.time
    )
    + arim.signal.Hilbert()
)
frame = frame.expand_frame_assuming_reciprocity()
probe_p = frame.probe.to_oriented_points()


# %% Plot interfaces

aplt.plot_interfaces(
    [probe_p, *frame.examination_object.walls.values()],
    show_orientations=False,
    show_last=True,
    markers=["."] + ["-"] * len(frame.examination_object.walls),
    filename=str(result_dir / "interfaces"),
    savefig=save,
)

# %% Ray tracing
grid = arim.io.grid_from_conf(conf)
grid = g.MaskedGrid.mask_grid_where(grid, (grid.z > 26e-3) & (grid.x < 28.5e-3))
grid_p = grid.to_oriented_points()

views = bic.make_views(
    frame.examination_object,
    probe_p,
    grid_p,
    tfm_unique_only=True,
    walls_for_imaging=imaging_walls,
)

arim.ray.ray_tracing(
    views.values(),
    convert_to_fortran_order=True,
    walls=frame.examination_object.walls,
    turn_off_invalid_rays=False,
)

# %% Compute TFM

tfms = dict()
for viewname, view in views.items():
    with arim.helpers.timeit(f"TFM {view.name}"):
        amplitudes = None  # arim.im.tfm.TxRxAmplitudes(
        #     (~view.tx_path.rays._invalid_rays.T).astype(float),
        #     (~view.rx_path.rays._invalid_rays.T).astype(float),
        # )
        tfms[viewname] = arim.im.tfm.tfm_for_view(
            frame,
            grid,
            view,
            fillvalue=0.0,
            interpolation="nearest",
            amplitudes=amplitudes,
        )

# %% Plot TFM

reference_rect = conf.get("reference_rect", None)
if reference_rect is None:
    reference_area = None
else:
    reference_area = grid.points_in_rectbox(**reference_rect)


# dynamic dB scale:
scale = aplt.common_dynamic_db_scale(
    [tfm.res for tfm in tfms.values()], reference_area, db_range=40.0
)

defect_centre = conf["scatterer"]["location"]

for i, ((viewname, tfm), (_, view)) in enumerate(zip(tfms.items(), views.items())):
    assert tfm.grid is grid

    ref_db, clim = next(scale)

    if reference_rect is None:
        patches = None
    else:
        patches = [
            mpl.patches.Rectangle(
                (reference_rect["xmin"], reference_rect["zmin"]),
                reference_rect["xmax"] - reference_rect["xmin"],
                reference_rect["zmax"] - reference_rect["zmin"],
                fill=False,
                edgecolor="magenta",
            )
        ]

    ax, _ = aplt.plot_tfm(
        tfm,
        clim=clim,
        scale="db",
        ref_db=ref_db,
        title=f"TFM {view.longname}",
        savefig=False,
        patches=patches,
        draw_cbar=True,
        interpolation="none",
    )
    aplt.plot_interfaces(
        [probe_p, *frame.examination_object.walls.values()],
        markers=["."] + ["r-"] * len(frame.examination_object.walls),
        ax=ax,
        show_legend=False,
        title=None,
    )
    ax.plot(defect_centre["x"], defect_centre["z"], "ow")
    ax.set_adjustable("box")
    ax.axis([grid.xmin, grid.xmax, grid.zmax, 0])
    if save:
        ax.figure.savefig(str(result_dir / f"tfm_{i:02}_{view.longname}"))

# Block script until windows are closed.
plt.show()
