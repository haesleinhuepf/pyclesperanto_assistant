from dataclasses import dataclass, field
from functools import partial
from types import ModuleType
from typing import Any, Sequence, Tuple, Type

import numpy as np
from napari.layers import Image, Labels, Layer
from typing_extensions import Annotated

FloatRange = Annotated[float, {"min": np.finfo(np.float32).min, "max": np.finfo(np.float32).max}]
PositiveFloatRange = Annotated[float, {"min": 0, "max": np.finfo(np.float32).max}]
PositiveIntRange = Annotated[int, {"min": 0, "max": 65535}]
ImageInput = Annotated[Image, {"label": "Image"}]
LayerInput = Annotated[Layer, {"label": "Image"}]
LabelsInput = Annotated[Labels, {"label": "Labels"}]
global_magic_opts = {"auto_call": True}

import pyclesperanto_prototype as cle


def nop(source):
    return source


@dataclass
class Category:
    name: str
    inputs: Sequence[Type]
    default_op: str
    output: str = "image"  # or labels
    # [(name, annotation, default), ...]
    args: Sequence[Tuple[str, Type, Any]] = field(default_factory=tuple)
    # categories
    include: Sequence[str] = field(default_factory=tuple)
    exclude: Sequence[str] = field(default_factory=tuple)
    # visualization
    color_map : str = "gray"
    blending : str = "translucent"
    # explicit module / function
    operations : Sequence[str] = None
    module : ModuleType = cle
    transfer_to : partial = partial(cle.push)
    transfer_from : partial = partial(nop)


CATEGORIES = {
    "Noise removal": Category(
        name="Noise removal",
        inputs=(ImageInput,),
        default_op="gaussian_blur",
        args=[
            ("x", FloatRange, 1),
            ("y", FloatRange, 1),
            ("z", FloatRange, 0)
        ],
        include=("filter", "denoise"),
        exclude=("combine",),
    ),
    "Background removal": Category(
        name="Background removal",
        inputs=(ImageInput,),
        default_op="top_hat_box",
        args=[
            ("x", FloatRange, 10),
            ("y", FloatRange, 10),
            ("z", FloatRange, 0)
        ],
        include=("filter", "background removal"),
        exclude=("combine",),
    ),
    "Filter": Category(
        name="Filter",
        inputs=(ImageInput,),
        default_op="gamma_correction",
        args=[
            ("x", FloatRange, 1),
            ("y", FloatRange, 1),
            ("z", FloatRange, 0)
        ],
        include=("filter",),
        exclude=("combine", "denoise", "background removal"),
    ),
    "Combine": Category(
        name="Combine",
        inputs=(LayerInput, LayerInput),
        default_op="add_images",
        include=("combine",),
        exclude=("map",),
        args=[
            ("a", FloatRange, 1),
            ("b", FloatRange, 1),
        ]
    ),
    "Transform": Category(
        name="Transform",
        inputs=(LayerInput,),
        default_op="sub_stack",
        output="image",  # can also be labels
        args=[
            ("a", FloatRange, 0),
            ("b", FloatRange, 0),
            ("c", FloatRange, 0),
            ("d", bool, False),
            ("e", bool, False),
        ],
        include=("transform",),
    ),
    "Projection": Category(
        name="Projection",
        inputs=(LayerInput,),
        default_op="maximum_z_projection",
        output="image",  # can also be labels
        include=("projection",),
    ),
    "Binarize": Category(
        name="Binarize",
        inputs=(LayerInput,),
        default_op="threshold_otsu",
        output="labels",
        args=[
            ("radius_x", PositiveFloatRange, 1),
            ("radius_y", PositiveFloatRange, 1),
            ("radius_z", PositiveFloatRange, 0),
        ],
        include=("binarize",),
        exclude=("combine",),
    ),
    "Label": Category(
        name="Label",
        inputs=(LayerInput,),
        default_op="connected_components_labeling_box",
        output="labels",
        args=[
            ("a", PositiveFloatRange, 2),
            ("b", PositiveFloatRange, 2)
        ],
        include=("label",),
    ),
    "Label processing": Category(
        name="Label processing",
        inputs=(LabelsInput,),
        default_op="exclude_labels_on_edges",
        output="labels",
        args=[
            ("min", PositiveFloatRange, 2),
            ("max", PositiveFloatRange, 100)
        ],
        include=("label processing",),
    ),
    "Label measurements": Category(
        name="Label measurements",
        inputs=(ImageInput, LabelsInput),
        default_op="label_mean_intensity_map",
        args=[
            ("n", PositiveFloatRange, 1)
        ],
        include=("combine", "map"),
        color_map="turbo",
        blending="translucent",
    ),
    "Map": Category(
        name="Map",
        inputs=(LabelsInput,),
        default_op="label_pixel_count_map",
        args=[
            ("n", PositiveFloatRange, 1),
            ("m", PositiveFloatRange, 1)
        ],
        include=("label measurement", "map"),
        exclude=("combine",),
        color_map="turbo",
        blending="translucent",
    ),
    "Mesh": Category(
        name="Mesh",
        inputs=(LabelsInput,),
        default_op="draw_mesh_between_touching_labels",
        args=[
            ("n", PositiveFloatRange, 1)
        ],
        include=("label measurement", "mesh"),
        color_map="green",
        blending="additive",
    )
}

from skimage import filters

CATEGORIES["skimage filters"] = Category(
        name="skimage filters",
        inputs=(ImageInput,),
        default_op="gaussian",
        args=[
            ("sigma", PositiveFloatRange, 1)
        ],
        operations=["gaussian", "hessian"],
        module=filters,
        transfer_to=partial(np.asarray),
        transfer_from=partial(nop)
    )

from scipy import ndimage

CATEGORIES["scipy ndimage filters"] = Category(
        name="scipy ndimage filters",
        inputs=(ImageInput,),
        default_op="median_filter",
        args=[
            ("size", PositiveIntRange, 1)
        ],
        operations=["maximum_filter", "median_filter", "minimum_filter"],
        module=ndimage,
        transfer_to=partial(np.asarray),
        transfer_from=partial(nop)
    )

try:
    import cupy
    from cupyx.scipy import ndimage as cdi

    CATEGORIES["cupy filters"] = Category(
        name="cupy filters",
        inputs=(ImageInput,),
        default_op="median_filter",
        args=[
            ("size", PositiveIntRange, 1)
        ],
        operations=["maximum_filter", "median_filter", "minimum_filter"],
        module=cdi,
        transfer_to=partial(cupy.asarray),
        transfer_from=partial(cupy.asnumpy)
    )

except ModuleNotFoundError:
    print("Cancelled setup cupy; not installed")
except ImportError:
    print("Cancelled setup cupy; not installed")
