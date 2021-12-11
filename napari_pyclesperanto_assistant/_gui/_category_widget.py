from __future__ import annotations

from inspect import Parameter, Signature, signature

from qtpy import QtCore
from typing import Any, Optional, TYPE_CHECKING, Sequence

import pyclesperanto_prototype as cle
import toolz
from loguru import logger
from magicgui import magicgui
from typing_extensions import Annotated
import napari

from .._categories import Category
from qtpy.QtWidgets import QPushButton, QDockWidget

if TYPE_CHECKING:
    from napari.layers import Layer
    from napari import Viewer

VIEWER_PARAM = "viewer"
OP_NAME_PARAM = "op_name"
OP_ID = "op_id"

from .._categories import FloatRange, BoolType, StringType
category_args = [
    ("x", FloatRange, 10),
    ("y", FloatRange, 10),
    ("z", FloatRange, 0),
    ("u", FloatRange, 0),
    ("v", FloatRange, 0),
    ("w", FloatRange, 0),
    ("a", BoolType, True),
    ("b", BoolType, True),
    ("c", BoolType, True),
    ("k", StringType, ""),
    ("l", StringType, ""),
    ("m", StringType, ""),
]
category_args_numeric = ["x", "y", "z", "u", "v", "w"]
category_args_bool = ["a", "b", "c"]
category_args_text = ["k", "l", "m"]

def num_positional_args(func, types=[cle.Image, int, str, float, bool]) -> int:
    params = signature(func).parameters
    return len([p for p in params.values() if p.annotation in types])


@logger.catch
def call_op(op_name: str, inputs: Sequence[Layer], timepoint : int = None, viewer: napari.Viewer = None, **kwargs) -> cle.Image:
    """Call cle operation `op_name` with specified inputs and args.

    Takes care of transfering data to GPU and omitting extra positional args

    Parameters
    ----------
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
    if timepoint is None:
        i0 = inputs[0].data
        gpu_ins = [i.data if i is not None else i0 for i in inputs]
    else:
        i0 = inputs[0].data[timepoint] if len(inputs[0].data.shape) == 4 else inputs[0].data
        gpu_ins = [(i.data[timepoint] if len(i.data.shape) == 4 else i.data if i is not None else i0) for i in inputs]

    # convert 3d-1-slice-data into 2d data
    # to support 2d timelapse data
    gpu_ins = [i if len(i.shape) != 3 or i.shape[0] != 1 else i [0] for i in gpu_ins]

    # call actual cle function ignoring extra positional args
    cle_function = find_function(op_name)
    nargs = num_positional_args(cle_function)

    args = []
    new_sig = signature(cle_function)
    # get the names of positional parameters in the new operation
    param_names, numeric_param_names, bool_param_names, str_param_names = separate_argnames_by_type(
        new_sig.parameters.items())

    # go through all parameters and collect their values in an args-array
    num_count = 0
    str_count = 0
    bool_count = 0
    for key in param_names:
        if key in numeric_param_names:
            value = kwargs[category_args_numeric[num_count]]
            num_count = num_count + 1
        elif key in bool_param_names:
            value = kwargs[category_args_bool[bool_count]]
            bool_count = bool_count + 1
        elif key in str_param_names:
            value = kwargs[category_args_text[str_count]]
            str_count = str_count + 1
        args.append(value)
    args = tuple(args)

    if cle_function.__module__ == "pyclesperanto_prototype":
        # todo: we could make this a little faster by getting gpu_out from a central manager
        gpu_out = None

        logger.info(f"cle.{op_name}(..., {', '.join(map(str, args))})")
        args = ((*gpu_ins, gpu_out) + args)[:nargs]
        gpu_out = cle_function(*args)

        # return output
        return gpu_out, args
    else:
        args = (*gpu_ins, *args)[:nargs+1]
        #print("args", args)
        kwargs = {}

        import inspect
        sig = inspect.signature(cle_function)
        #for k, v in sig.parameters.items():
        #    print(k, v.annotation)
        #    if k == "viewer" or k == "napari_viewer" or "napari.viewer.Viewer" in str(v):
        #        kwargs[k] = viewer

        # Make sure that the annotated types are really passed to a given function
        for i, k in enumerate(list(sig.parameters.keys())):
            if i >= len(args):
                break
            type_annotation = str(sig.parameters[k].annotation)
            #print("Annotation:", type_annotation)
            args = list(args)
            for typ in ["int", "float", "str"]:
                if typ in type_annotation:
                    converter = eval(typ)
                    #print("converter", converter)
                    args[i] = converter(args[i])

        gpu_out = cle_function(*args, **kwargs)

        if sig.return_annotation in [napari.types.LabelsData, "napari.types.LabelsData"]:
            if gpu_out.dtype is not int:
                gpu_out = gpu_out.astype(int)

        return gpu_out, args
def find_function(op_name):
    cle_function = None
    try:
        cle_function = cle.operation(op_name)  # couldn't this just be getattr(cle, ...)?
    except:
        pass
    if cle_function is None:
        from .._categories import all_operations
        all_ops = all_operations()
        for k, f in all_ops.items():
            if op_name in k:
                cle_function = f
    if cle_function is None:
        print("No function found for", op_name)
    return cle_function

def _show_result(
    gpu_out: cle.Image,
    viewer: Viewer,
    name: str,
    layer_type: str,
    op_id: int,
    translate=None,
    cmap=None,
    blending=None,
    scale=None,
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
    #print("OP ID ", op_id)
    if not viewer:
        logger.warning("no viewer, cannot add image")
        return
    # show result in napari
    clims = [cle.minimum_of_all_pixels(gpu_out), cle.maximum_of_all_pixels(gpu_out)]

    if clims[1] == 0:
        clims[1] = 1

    # conversion will be done inside napari. We can continue working with the OCL-array from here.
    data = gpu_out

    try:
        # look for an existing layer
        layer = next(x for x in viewer.layers if isinstance(x.metadata, dict) and x.metadata.get(OP_ID) == op_id)
        logger.debug(f"updating existing layer: {layer}, with id: {op_id}")
        layer.data = data
        layer.name = name
        # layer.translate = translate
    except StopIteration:
        # otherwise create a new one
        logger.debug(f"creating new layer for id: {op_id}")
        add_layer = getattr(viewer, f"add_{layer_type}")
        kwargs = dict(name=name, metadata={OP_ID: op_id})
        if layer_type == "image":
            kwargs["colormap"] = cmap
            kwargs["blending"] = blending
            kwargs['contrast_limits'] = clims
        layer = add_layer(data, **kwargs)

    if scale is not None:
        if len(layer.data.shape) <= len(scale):
            layer.scale = scale[-len(layer.data.shape):]
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
    from .._categories import operations_in_menu
    choices = list(operations_in_menu(category.tools_menu))
    #print("choices:", choices)
    op_type = Annotated[str, {"choices": choices, "label": "Operation"}]
    default_op = category.default_op
    if not any(default_op == op for op in choices):
        #print("Default-operation is not in list!")
        default_op = None

    if default_op is None:
        params.append(
            Parameter(OP_NAME_PARAM, k, annotation=op_type)
        )
    else:
        params.append(
            Parameter(OP_NAME_PARAM, k, annotation=op_type, default=default_op)
        )
    # add the args that will be passed to the cle operation.
    for name, type_, default in category_args:
        params.append(Parameter(name, k, annotation=type_, default=default))

    # add a viewer.  This allows our widget to know if it's in a viewer
    params.append(
        Parameter(VIEWER_PARAM, k, annotation="napari.viewer.Viewer", default=None)
    )
    result = Signature(params)
    #print("Signature", result)
    return result


def make_gui_for_category(category: Category, viewer: napari.Viewer = None) -> magicgui.widgets.FunctionGui[Layer]:
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
    widget = None
    def gui_function(**kwargs) -> Optional[Layer]:
        """A function that calls a cle operation `call_op` and shows the result.

        This is the function that will be called by our magicgui widget.
        We modify it's __signature__ below.
        """
        viewer = kwargs.pop(VIEWER_PARAM, None)
        inputs = [kwargs.pop(k) for k in list(kwargs) if k.startswith("input")]
        t_position = None
        if viewer is not None and len(viewer.dims.current_step) == 4:
            # in case we process a 4D-data set, we need read out the current timepoint
            # and consider it further down in call_op
            t_position = viewer.dims.current_step[0]

            currstep_event = viewer.dims.events.current_step

            def update(event):
                currstep_event.disconnect(update)
                widget()

            if hasattr(widget, 'updater'):
                currstep_event.disconnect(widget.updater)

            widget.updater = update

            currstep_event.connect(update)

        # todo: deal with 5D and nD data
        op_name = kwargs.pop("op_name")
        result, used_args = call_op(op_name, inputs, t_position, viewer, **kwargs)

        # add a help-button
        description = find_function(op_name).__doc__
        if description is not None:
            description = description.replace("\n    ", "\n") + "\n\nRight-click to learn more..."
            temp = description.split('https:')
            link = "https://napari-hub.org/plugins/napari-pyclesperanto-assistant"
            if len(temp) > 1:
                link = "https:" + temp[1].split("\n")[0]
            getattr(widget, OP_NAME_PARAM).native.setToolTip(description)

        # Right-click: Open online help
        #combobox = getattr(widget, OP_NAME_PARAM).native
        #combobox.orig_mousePressEvent = getattr(widget, OP_NAME_PARAM).native.mousePressEvent
        #def call_link(event):
        #    if event.button() == QtCore.Qt.RightButton:
        #        import webbrowser
        #        webbrowser.open(link)
        #    else:
        #        combobox.orig_mousePressEvent(event)
        #combobox.mousePressEvent = call_link

        if result is not None:
            result_layer = _show_result(
                result,
                viewer,
                name=f"Result of {op_name}",
                layer_type=category.output,
                op_id=id(gui_function),
                cmap=category.color_map,
                blending=category.blending,
                scale=inputs[0].scale,
            )

            # notify workflow manage that something was created / updated
            try:
                from napari_time_slicer import WorkflowManager
                manager = WorkflowManager.install(viewer)
                manager.update(result_layer, find_function(op_name), *used_args)
                #print("notified", result_layer.name, find_function(op_name))
            except ImportError:
                pass # recording workflows in the WorkflowManager is a nice-to-have at the moment.

            def _on_layer_removed(event):
                layer = event.value
                if layer in inputs or layer is result_layer:
                    try:
                        viewer.window.remove_dock_widget(widget.native)
                    except:
                        pass

            viewer.layers.events.removed.connect(_on_layer_removed)

            return result_layer
        return None

    gui_function.__name__ = f'do_{category.name.lower().replace(" ", "_")}'
    gui_function.__signature__ = _generate_signature_for_category(category)

    # create the widget
    widget = magicgui(gui_function, auto_call=True)

    # when the operation name changes, we want to update the argument labels
    # to be appropriate for the corresponding cle operation.
    op_name_widget = getattr(widget, OP_NAME_PARAM)

    @op_name_widget.changed.connect
    def update_positional_labels(*_: Any):
        func = find_function(op_name_widget.value)
        new_sig = signature(func)
        # get the names of positional parameters in the new operation
        param_names, numeric_param_names, bool_param_names, str_param_names = separate_argnames_by_type(
            new_sig.parameters.items())

        print("function", func.__name__)
        print("numeric_param_names", numeric_param_names)

        num_count = 0
        str_count = 0
        bool_count = 0

        # update the labels of each positional-arg subwidget

        n_params = len(param_names)


        # show needed elements and set right label
        for n, arg in enumerate(category_args):
            arg_gui_name = arg[0]
            arg_gui_type = arg[1]
            wdg = getattr(widget, arg_gui_name)
            if arg_gui_type == FloatRange:
                if num_count < len(numeric_param_names):
                    arg_func_name = numeric_param_names[num_count]
                    num_count = num_count + 1
                else:
                    wdg.hide()
                    continue
            elif arg_gui_type == BoolType:
                if bool_count < len(bool_param_names):
                    arg_func_name = bool_param_names[bool_count]
                    bool_count = bool_count + 1
                else:
                    wdg.hide()
                    continue
            elif arg_gui_type == StringType:
                if str_count < len(str_param_names):
                    arg_func_name = str_param_names[str_count]
                    str_count = str_count + 1
                else:
                    wdg.hide()
                    continue
            else:
                arg_func_name = "?"
                print("Unsupported type:", arg_gui_type)
                continue

            wdg.label = arg_func_name
            wdg.text = arg_func_name
            wdg.show()

    # run it once to update the labels
    update_positional_labels()

    return widget


def separate_argnames_by_type(items):
    param_names = [
        name
        for name, param in items
        if param.annotation in {int, str, float, bool}
    ]
    numeric_param_names = [
        name
        for name, param in items
        if param.annotation in {int, float}
    ]
    bool_param_names = [
        name
        for name, param in items
        if param.annotation in {bool}
    ]
    str_param_names = [
        name
        for name, param in items
        if param.annotation in {str}
    ]
    return param_names, numeric_param_names, bool_param_names, str_param_names
