"""
Extraction of scattering matrix using generalised imaging.
"""

from arim.geometry import Grid, Points, default_oriented_points
import arim.im
import arim.io
import arim.models.block_in_contact as bic
import arim.plot as aplt
from arim.signal import Hanning, Hilbert
import numpy as np


# %% Load frame
conf = arim.io.load_conf(".")
frame = arim.io.frame_from_conf(conf, use_probe_from_conf=True)
frame = frame.apply_filter(
    Hanning(frame.numsamples, frame.probe.frequency, frame.probe.frequency, frame.time)
    + Hilbert()
)
frame = frame.expand_frame_assuming_reciprocity()

# %% TFM imaging
grid = arim.io.grid_from_conf(conf)

tfm_views = bic.make_views(
    frame.examination_object,
    frame.probe.to_oriented_points(),
    grid.to_oriented_points(),
    tfm_unique_only=True,
)
views_to_use = conf.get("views_to_use", "all")
if views_to_use != "all":
    tfm_views = {
        viewname: view
        for viewname, view in tfm_views.items()
        if viewname in views_to_use
    }
arim.ray.ray_tracing(tfm_views.values())

tfms = dict()
for viewname, view in tfm_views.items():
    tfms[viewname] = arim.im.tfm.tfm_for_view(
        frame, grid, view, fillvalue=0.0, interpolation="nearest"
    )

aplt.plot_tfm(tfms["L - L"], func_res=arim.ut.decibel)

# %% Generalised imaging
# Make a grid around the known scatterer location.
point = Points(
    [
        [
            20.4e-3,
            0.0,
            32.4e-3,
        ]
    ]
)
generalised_grid = Grid(
    xmin=point.x[0] - 1.725e-3,
    xmax=point.x[0] + 1.725e-3,
    ymin=0.0,
    ymax=0.0,
    zmin=point.z[0] - 1.725e-3,
    zmax=point.z[0] + 1.725e-3,
    pixel_size=(
        1 / 5 * tfm_views["L - L"].tx_path.velocities[-1] / frame.probe.frequency,
        0.0,
        1 / 20 * tfm_views["L - L"].tx_path.velocities[-1] / frame.probe.frequency,
    ),
)

point_views = bic.make_views(
    frame.examination_object,
    frame.probe.to_oriented_points(),
    default_oriented_points(point),
    tfm_unique_only=True,
)
generalised_views = bic.make_views(
    frame.examination_object,
    frame.probe.to_oriented_points(),
    generalised_grid.to_oriented_points(),
    tfm_unique_only=True,
)
views_to_use = conf.get("views_to_use", "all")
if views_to_use != "all":
    point_views = {
        viewname: view
        for viewname, view in point_views.items()
        if viewname in views_to_use
    }
    generalised_views = {
        viewname: view
        for viewname, view in generalised_views.items()
        if viewname in views_to_use
    }
arim.ray.ray_tracing(point_views.values())
arim.ray.ray_tracing(generalised_views.values())

model_options = dict(
    probe_element_width=np.linalg.norm(np.diff(frame.probe.locations[:2], axis=0))
)
weights = bic.ray_weights_for_views(point_views, frame.probe.frequency, **model_options)

generalised_images = dict()
inverse_images = dict()
for (viewname, point_view), (_, grid_view) in zip(
    point_views.items(), generalised_views.items()
):
    generalised_images[viewname] = arim.im.generalised.generalised_image_for_view(
        frame,
        generalised_grid,
        grid_view,
        fillvalue=0.0,
        interpolation="nearest",
        offdiagonallength=2,
    )
    inverse_images[viewname] = arim.im.generalised.inverse_for_generalised_image(
        generalised_images[viewname],
        point_view,
        grid_view,
        6 / frame.probe.frequency,
        frame,
    )
    spec = np.fft.fft(inverse_images[viewname].timetraces, axis=1)
