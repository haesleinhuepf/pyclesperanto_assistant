from __future__ import annotations

from inspect import Parameter, Signature, signature
from typing import Optional, TYPE_CHECKING, Sequence

import pyclesperanto_prototype as cle
from loguru import logger
from magicgui import magicgui
from typing_extensions import Annotated
import numpy as np

from .._categories import Category

if TYPE_CHECKING:
    from napari.layers import Layer
    from napari import Viewer

VIEWER_PARAM = "viewer"
OP_NAME_PARAM = "op_name"
OP_ID = "op_id"

def num_positional_args(func, types=[cle.Image, int, str, float, bool]) -> int:
    params = signature(func).parameters
    return len([p for p in params.values() if p.annotation in types])


@logger.catch
def call_op(category : Category, op_name: str, inputs: Sequence[Layer], *args) -> cle.Image:
    """Call cle operation `op_name` with specified inputs and args.

    Takes care of transfering data to GPU and omitting extra positional args

    Parameters
    ----------
    category : Category
        category the function belongs to
    op_name : str
        name of operation to execute.  (must be valid for `cle.operation`)
    inputs : Sequence[Layer]
        The napari layer inputs

    Returns
    -------
    cle.Image
        The result (still on the GPU)
    """

    if not inputs or inputs[0] is None:
        return

    # transfer data to gpu
    i0 = inputs[0].data
    #if module == cle:
    #    gpu_ins = [cle.push(i.data if i is not None else i0) for i in inputs]
    #else:
    #    gpu_ins = [np.asarray(i.data) if i is not None else np.asarray(i0) for i in inputs]
    gpu_ins = [category.transfer_to(i.data) if i is not None else category.transfer_to(i0) for i in inputs]

    # todo: we could make this a little faster by getting gpu_out from a central manager
    gpu_out = None

    # call actual function ignoring extra positional args
    cle_function = getattr(category.module, op_name)
    nargs = num_positional_args(cle_function)
    logger.info(f"cle.{op_name}(..., {', '.join(map(str, args))})")
    args = (*gpu_ins, gpu_out) + args
    if category.module == cle: # todo get rid of cle here
        args = args[:nargs]
    else:
        args = [x for x in args if x is not None]

    gpu_out = cle_function(*args)

    # return output
    return gpu_out


def _show_result(
    gpu_out: cle.Image,
    viewer: Viewer,
    name: str,
    layer_type: str,
    op_id: int,
    translate=None,
    cmap=None,
    blending=None,
    category:Category = None
) -> Optional[Layer]:
    """Show `gpu_out` in the napari viewer.

    Parameters
    ----------
    gpu_out : cle.Image
        a cle.Image to show
    viewer : napari.Viewer
        The napari viewer instance
    name : str
        The name of the layer to create or update.
    layer_type : str
        the layer type to create (must be 'labels' or 'image)
    op_id : int
        an ID to associate with the newly created layer (will be added to
        layer.metada['op_id'])
    translate : [type], optional
        translate parameter for layer creation, by default None
    cmap : str, optional
        a colormap to use for images, by default None
    blending : str, optional
        blending mode for visualization, by default None

    Returns
    -------
    layer : Optional[Layer]
        The created/udpated layer, or None if no viewer is present.
    """
    if not viewer:
        logger.warning("no viewer, cannot add image")
        return
    # show result in napari
    if isinstance(gpu_out, cle._tier0._pycl.OCLArray): # todo: don't refer to cle internals
        clims = [cle.minimum_of_all_pixels(gpu_out), cle.maximum_of_all_pixels(gpu_out)]

        if clims[1] == 0:
            clims[1] = 1
    else:
        clims = None

    # conversion will be done inside napari. We can continue working with the OCL-array from here.
    data = category.transfer_from(gpu_out)

    try:
        # look for an existing layer
        layer = next(x for x in viewer.layers if x.metadata.get(OP_ID) == op_id)
        logger.debug(f"updating existing layer: {layer}, with id: {op_id}")
        layer.data = data
        layer.name = name
        if layer_type != "labels" and clims:
            layer.contrast_limits = clims
        # layer.translate = translate
    except StopIteration:
        # otherwise create a new one
        logger.debug(f"creating new layer for id: {op_id}")
        add_layer = getattr(viewer, f"add_{layer_type}")
        kwargs = dict(name=name, metadata={OP_ID: op_id})
        if layer_type == "image":
            kwargs["colormap"] = cmap
            kwargs["blending"] = blending
        layer = add_layer(data, **kwargs)
    return layer


def _generate_signature_for_category(category: Category) -> Signature:
    """Create an inspect.Signature object representing a cle Category.

    The output of this function can be used to set function.__signature__ so that
    magicgui can convert it into the appropriate widget.
    """
    k = Parameter.KEYWORD_ONLY

    # add inputs (we name them inputN ...)
    params = [
        Parameter(f"input{n}", k, annotation=t) for n, t in enumerate(category.inputs)
    ]
    # Add valid operations choices (will create the combo box)
    if category.operations:
        choices = category.operations
    else:
        choices = list(cle.operations(['in assistant'] + list(category.include), category.exclude))
    op_type = Annotated[str, {"choices": choices, "label": "Operation"}]
    params.append(
        Parameter(OP_NAME_PARAM, k, annotation=op_type, default=category.default_op)
    )
    # add the args that will be passed to the cle operation.
    for name, type_, default in category.args:
        params.append(Parameter(name, k, annotation=type_, default=default))

    # add a viewer.  This allows our widget to know if it's in a viewer
    params.append(
        Parameter(VIEWER_PARAM, k, annotation="napari.viewer.Viewer", default=None)
    )
    return Signature(params)


def make_gui_for_category(category: Category) -> magicgui.widgets.FunctionGui[Layer]:
    """Generate a magicgui widget for a Category object

    Parameters
    ----------
    category : Category
        An instance of a _categories.Category. (holds information about the cle operations,
        input types, and arguments that the widget needs to represent.)

    Returns
    -------
    magicgui.widgets.FunctionGui
        A magicgui widget instance
    """

    def gui_function(**kwargs) -> Optional[Layer]:
        """A function that calls a cle operation `call_op` and shows the result.

        This is the function that will be called by our magicgui widget.
        We modify it's __signature__ below.
        """
        viewer = kwargs.pop(VIEWER_PARAM, None)
        inputs = [kwargs.pop(k) for k in list(kwargs) if k.startswith("input")]
        op_name = kwargs.pop("op_name")
        result = call_op(category, op_name, inputs, *kwargs.values())
        if result is not None:
            return _show_result(
                result,
                viewer,
                name=f"Result of {op_name}",
                layer_type=category.output,
                op_id=id(gui_function),
                cmap=category.color_map,
                blending=category.blending,
                category=category
            )
        return None

    gui_function.__name__ = f'do_{category.name.lower().replace(" ", "_")}'
    gui_function.__signature__ = _generate_signature_for_category(category)

    # create the widget
    widget = magicgui(gui_function, auto_call=True)

    # when the operation name changes, we want to update the argument labels
    # to be appropriate for the corresponding cle operation.
    op_name_widget = getattr(widget, OP_NAME_PARAM)

    @op_name_widget.changed.connect
    def update_positional_labels(*_):
        if category.module != cle:
            return
        new_sig = signature(getattr(category.module, op_name_widget.value))
        # get the names of positional parameters in the new operation
        param_names = [
            name
            for name, param in new_sig.parameters.items()
            if param.annotation in {int, str, float, bool}
        ]

        # update the labels of each positional-arg subwidget
        # or, if there are too many, hide them
        n_params = len(param_names)
        for n, arg in enumerate(category.args):
            wdg = getattr(widget, arg[0])
            if n < n_params:
                wdg.label = param_names[n]
                wdg.text = param_names[n]
                wdg.show()
            else:
                wdg.hide()

    # run it once to update the labels
    update_positional_labels()

    return widget
