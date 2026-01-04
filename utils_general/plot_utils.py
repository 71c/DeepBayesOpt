import sys
from typing import Any, List, Optional, Sequence, Set, TypeVar
import os
import logging
import warnings
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from linear_operator.utils.warnings import NumericalWarning

from utils_general.io_utils import save_json
from utils_general.utils import dict_to_str


def add_plot_args(parser):
    plot_group = parser.add_argument_group("Plotting organization")
    plot_group.add_argument(
        '--use_cols',
        action='store_true',
        help='Whether to use columns for subplots in the plots'
    )
    plot_group.add_argument(
        '--use_rows',
        action='store_true',
        help='Whether to use rows for subplots in the plots'
    )
    plot_group.add_argument(
        '--plots_group_name',
        type=str,
        help='Name of group of plots',
    )
    plot_group.add_argument(
        '--plots_name',
        type=str,
        help='Name of these plots'
    )


def _count_num_plots(plot_config: dict, level_names: list[str], all_seeds=True):
    # Count the number of plots in the plot_config
    next_level_names = level_names[1:] if len(level_names) > 1 else []
    n_plots = 0
    for k, v in plot_config.items():
        items = v["items"]

        if all_seeds:
            if len(next_level_names) >= 1 and next_level_names[0] == "line":
                n_plots += 1
            else:
                if isinstance(items, dict):
                    n_plots += _count_num_plots(
                        items, next_level_names, all_seeds=all_seeds)
                else:
                    n_plots += 1
        else:
            # TODO: handle non-all_seeds case properly with level_names (if desired)
            if isinstance(items, dict):
                itemss = [v["items"] for v in items.values()]
                if all(isinstance(i, dict) for i in itemss):
                    n_plots += _count_num_plots(
                        items, next_level_names, all_seeds=all_seeds)
                elif any(isinstance(i, dict) for i in itemss):
                    raise ValueError("Invalid plot config")
                else:
                    n_plots += 1
            else:
                raise RuntimeError("This should not happen")
    return n_plots


def _add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=30,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


def _get_figure_from_nested_structure(
        plot_config: dict,
        plot_ax_func,
        attr_name_to_title: dict[str, str],
        attrs_groups_list: list[Optional[set]],
        level_names: list[str],
        figure_name: str,
        pbar=None,
        special_names=set(),
        kv_to_str_func=None,
        **plot_kwargs):
    try:
        this_attrs_group = attrs_groups_list[0]
        next_attrs_groups = attrs_groups_list[1:]
    except IndexError:
        this_attrs_group = set()
        next_attrs_groups = [set()]

    # Ensure next_attrs_groups is not empty
    if not next_attrs_groups:
        next_attrs_groups = [set()]

    try:
        this_level_name = level_names[0]
        next_level_names = level_names[1:]
    except IndexError:
        this_level_name = "line"

    scale = plot_kwargs.get("scale", 1.0)
    aspect = plot_kwargs.get("aspect", 1.5)
    sharey = plot_kwargs.get("sharey", False)

    area = 50 * scale**2
    height = np.sqrt(area / aspect)
    width = aspect * height

    row_and_col = False
    col_names = None

    next_attrs_groups_0 = next_attrs_groups[0] if len(next_attrs_groups) > 0 else set()

    # Handle None values -- hacky fix, not sure if it's what I want
    if next_attrs_groups_0 is None:
        next_attrs_groups_0 = set()
    if this_attrs_group is None:
        this_attrs_group = set()

    tmp_this = any(x in this_attrs_group for x in special_names)
    tmp_next = any(x in next_attrs_groups_0 for x in special_names)

    if this_level_name == "line":
        n_rows = 1
        n_cols = 1

        # irrelevant:
        sharey = False
        sharex = False
    elif this_level_name == "row":
        n_rows = len(plot_config)
        if next_level_names[0] == "col":
            row_and_col = True
            assert next_level_names[1] == "line"
            key_to_data = {}
            for v in plot_config.values():
                for kk, vv in v["items"].items():
                    if kk not in key_to_data:
                        tmp = [
                            (a,
                             0 if b is None or b == "None" else 1,
                             b)
                            for a, b in vv["vals"].items()
                        ]
                        key_to_data[kk] = list(sorted(tmp))

            col_names = list(sorted(key_to_data.keys(),
                                    key=lambda u: key_to_data[u]))

            col_name_to_col_index = {}
            for i, col_name in enumerate(col_names):
                col_name_to_col_index[col_name] = i
            n_cols = len(col_names)
            if tmp_this:
                if tmp_next:
                    sharey = False
                    sharex = False
                else:
                    # The plot attribute is varied with each row.
                    sharey = "row" # each subplot row will share an x- or y-axis.
                    sharex = "row"
            elif tmp_next:
                # The plot attribute is varied with each column.
                sharey = "col" # each subplot column will share an x- or y-axis.
                sharex = "col"
            else:
                sharey = True
                sharex = True
        else:
            n_cols = 1
            sharey = not tmp_this
            sharex = not tmp_this
    elif this_level_name == "col":
        n_rows = 1
        n_cols = len(plot_config)
        sharey = not tmp_this
        sharex = not tmp_this
    else:
        raise ValueError

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(width * n_cols, height * n_rows),
                             sharex=sharex, sharey=sharey, squeeze=False)

    def _plot_ax_func(*args, **kwargs):
        plot_ax_func(*args, **kwargs)
        if pbar is not None:
            pbar.update(1)

    if this_level_name == "line":
        _plot_ax_func(plot_config=plot_config, ax=axes[0, 0], plot_name=figure_name,
                      attr_name_to_title=attr_name_to_title, **plot_kwargs)
    else:
        fig.suptitle(figure_name, fontsize=16, fontweight='bold')

        sorted_plot_config_items = list(sorted(
            plot_config.items(),
            key=lambda x: sorted(
                [
                    (a,
                     kv_to_str_func(a, b)[0], # priority
                     0 if b is None or b == "None" else 1,
                     b)
                    for a, b in x[1]["vals"].items()
                ]
            )
        ))

        if row_and_col:
            for row, (row_name, row_data) in enumerate(sorted_plot_config_items):
                row_items = row_data["items"]
                for subplot_name, subplot_data in row_items.items():
                    col = col_name_to_col_index[subplot_name]
                    _plot_ax_func(plot_config=subplot_data["items"], ax=axes[row, col],
                                  plot_name=None, attr_name_to_title=attr_name_to_title,
                                  **plot_kwargs)
            row_names = [x[0] for x in sorted_plot_config_items]
            _add_headers(fig, row_headers=row_names, col_headers=col_names,
                        size='xx-large')
        elif this_level_name == "row" or this_level_name == "col":
            assert next_level_names[0] == "line"
            if this_level_name == "col":
                axs = axes[0, :]
            else:
                axs = axes[:, 0]

            row_names = [x[0] for x in sorted_plot_config_items]
            for ax, (subplot_name, subplot_data) in zip(axs, sorted_plot_config_items):
                _plot_ax_func(plot_config=subplot_data["items"], ax=ax,
                              plot_name=subplot_name,
                              attr_name_to_title=attr_name_to_title, **plot_kwargs)
        else:
            raise ValueError

    fig.tight_layout()

    return fig


def _save_figures_from_nested_structure(
        plot_config: dict,
        plot_ax_func,
        attrs_groups_list: list[Optional[set]],
        level_names: list[str],
        base_folder='',
        attr_name_to_title: dict[str, str] = {},
        pbar=None,
        special_names=set(),
        kv_to_str_func=None,
        **plot_kwargs):
    # Create the directory
    os.makedirs(base_folder, exist_ok=True)

    # Create a specific logger
    logger = logging.getLogger(base_folder)
    logger.setLevel(logging.WARNING)
    # Create a file handler for the logger
    file_handler = logging.FileHandler(os.path.join(base_folder, "warnings.log"))
    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the handler to the logger
    logger.addHandler(file_handler)

    with warnings.catch_warnings(record=True) as caught_warnings:
        # https://docs.python.org/3/library/warnings.html#the-warnings-filter
        # "default": print the first occurrence of matching warnings for each location
        # (module + line number) where the warning is issued
        warnings.simplefilter("default", category=NumericalWarning)

        this_attrs_group = attrs_groups_list[0]
        next_attrs_groups = attrs_groups_list[1:]
        this_level_name = level_names[0]
        next_level_names = level_names[1:]

        if this_attrs_group:
            save_json({"attrs": list(this_attrs_group)},
                        os.path.join(base_folder, "attrs.json"), indent=2)

        if this_level_name == "folder":
            for folder_name, data in plot_config.items():
                items = data["items"]
                dirname = os.path.join(base_folder, folder_name)

                if "vals" in data:
                    save_json(data["vals"], os.path.join(dirname, "vals.json"), indent=2)

                _save_figures_from_nested_structure(
                    items, plot_ax_func, next_attrs_groups, next_level_names,
                    base_folder=dirname,
                    attr_name_to_title=attr_name_to_title,
                    pbar=pbar,
                    special_names=special_names,
                    kv_to_str_func=kv_to_str_func,
                    **plot_kwargs
                )
        elif this_level_name == "fname":
            info_dict = {}
            for fname_desc, data in plot_config.items():
                items = data["items"]
                if "vals" in data:
                    info_dict[fname_desc] = data["vals"]

                # Wrap items in expected dictionary structure for the line level
                # The plotting function expects: {legend_name: {'items': data}}
                if isinstance(items, list):
                    # If items is a list of indices, wrap it in the expected structure
                    items_wrapped = {fname_desc: {'items': items}}
                    if "vals" in data:
                        items_wrapped[fname_desc]['vals'] = data['vals']
                else:
                    # If items is already a dict, use it as-is
                    items_wrapped = items

                fig = _get_figure_from_nested_structure(
                    items_wrapped, plot_ax_func, attr_name_to_title, next_attrs_groups,
                    next_level_names, fname_desc, pbar=pbar,
                    special_names=special_names, kv_to_str_func=kv_to_str_func,
                    **plot_kwargs)

                fname = f"{fname_desc}.pdf"
                fpath = os.path.join(base_folder, fname)
                fig.savefig(fpath, dpi=300, format='pdf', bbox_inches='tight')
                plt.close(fig)

            if info_dict:
                save_json(info_dict,
                        os.path.join(base_folder, "vals_per_figure.json"), indent=2)
        else:
            raise ValueError(f"Invalid level name: {this_level_name}")

        # Log the caught warnings using the specific logger
        for w in caught_warnings:
            logger.warning(
                warnings.formatwarning(w.message, w.category, w.filename, w.lineno))


def get_save_figures_from_nested_structure(special_names, kv_to_str_func):
    def ret(
            plot_config: dict,
            plot_ax_func,
            attrs_groups_list: list[Optional[set]],
            level_names: list[str],
            base_folder='',
            attr_name_to_title: dict[str, str] = {},
            print_pbar=True,
            all_seeds=True,
            **plot_kwargs):
        if type(plot_config) is tuple and len(plot_config) == 2 and \
            type(plot_config[0]) is dict and type(plot_config[1]) is list:
            plot_config = plot_config[0]
        elif type(plot_config) is not dict:
            print(f"{type(plot_config)=}", file=sys.stderr)
            if type(plot_config) in {list, tuple}:
                print(f"  len(plot_config)={len(plot_config)}", file=sys.stderr)
                for i, pc in enumerate(plot_config):
                    print(f"  {i}: {type(pc)=}", file=sys.stderr)
                    if type(pc) is dict:
                        print(f"    plot_config[{i}] type: dict with {len(pc)} keys",
                            file=sys.stderr)
                    elif type(pc) is list:
                        print(f"    plot_config[{i}] type: list with {len(pc)} items",
                            file=sys.stderr)
                        for j, item in enumerate(pc):
                            print(f"      item {j}: type {type(item)}, value {item}",
                                file=sys.stderr)
                    else:
                        print(f"    plot_config[{i}] type: {type(pc)}", file=sys.stderr)
            raise ValueError("plot_config should be a dict")

        n_plots = _count_num_plots(
            plot_config, level_names=level_names.copy(), all_seeds=all_seeds)
        pbar = tqdm(total=n_plots, desc="Saving figures") if print_pbar else None
        _save_figures_from_nested_structure(
            plot_config, plot_ax_func, attrs_groups_list, level_names,
            base_folder=base_folder,
            attr_name_to_title=attr_name_to_title,
            pbar=pbar,
            special_names=special_names,
            kv_to_str_func=kv_to_str_func,
            **plot_kwargs
        )
        if print_pbar:
            pbar.close()
    return ret


K = TypeVar('K')
V = TypeVar('V')


def _group_by_nested_attrs(items: List[dict[K, Any]],
                        attrs_groups_list: List[Set[K]],
                        dict_to_str_func,
                        return_single=False,
                        indices=None,
                        sort_key_for_grouped_items_func=None):
    if indices is None:
        indices = list(range(len(items)))

    if len(attrs_groups_list) == 0:
        if return_single and len(indices) == 1:
            return indices[0]
        return indices
    initial_attrs = attrs_groups_list[0]
    initial_grouped_items = {}
    to_add = []
    for idx in indices:
        item = items[idx]
        d = {k: item[k] for k in initial_attrs if k in item}

        # if len(d) == 0:
        #     raise ValueError(
        #         f"Got empty dictionary for plotting!\n{item=}\n{initial_attrs=}\n"
        #         "Must have forgotten to include an attribute in initial_attrs. "
        #         "Add it in the required place in registry.yml (yes it is annoying "
        #         "and manual).\nAlso by the way, if it hasn't been done already, "
        #         "consider adding formatting for the attribute in the function "
        #         "`plot_dict_to_str` in utils/plot_utils.py (if applicable).")

        # d = {k: v for k, v in d.items() if v is not None}
        d = {k: str(v) if v is None else v for k, v in d.items()}

        # if not d:
        #     # counts = group_by(
        #     #     item.keys(),
        #     #     lambda k: sum(other_item[k] != item[k]
        #     #                   for j, other_item in enumerate(items) if j != idx)
        #     # )
        #     # counts_sorted = sorted(counts.items(), reverse=True)
        #     # highest_mismatch_count, candidates =  counts_sorted[0]
        #     # key = str(candidates[0])
        #     # print("Found an item that would have an empty description. Using the "
        #     #       f"description {key} as it is different from the others "
        #     #       f"{highest_mismatch_count}/{len(items)} times.")
        #     key = ""

        key = dict_to_str_func(d)
        if set(d.keys()) == initial_attrs:
            if key in initial_grouped_items:
                initial_grouped_items[key]['items'].append(idx)
            else:
                initial_grouped_items[key] = {
                    'items': [idx],
                    'vals': d
                }
        else:
            to_add.append((key, idx, d))

    new_grouped_items = {}
    for key_to_add, idx_to_add, d_to_add in to_add:
        item_to_add = items[idx_to_add]
        count = 0
        for key, value in initial_grouped_items.items():
            if all(k not in item_to_add or item_to_add[k] == v for k, v in value['vals'].items()):
                initial_grouped_items[key]['items'].append(idx_to_add)
                count += 1
        if count == 0:
            if key_to_add in new_grouped_items:
                new_grouped_items[key_to_add]['items'].append(idx_to_add)
            else:
                new_grouped_items[key_to_add] = {
                    'items': [idx_to_add],
                    'vals': d_to_add
                }

    # initial_grouped_items = {key: value['items']
    #                         for key, value in initial_grouped_items.items()}
    initial_grouped_items.update(new_grouped_items)

    # Sort the grouped items by their values to ensure consistent legend ordering
    sorted_items = sorted(initial_grouped_items.items(),
                          key=sort_key_for_grouped_items_func)

    next_attrs = attrs_groups_list[1:]
    return {
        k: {
            'items': _group_by_nested_attrs(items, next_attrs, dict_to_str_func,
                                indices=v['items'], return_single=return_single,
                                sort_key_for_grouped_items_func=sort_key_for_grouped_items_func),
            'vals': v['vals']
        }
        for k, v in sorted_items
    }


def _are_all_disjoint(sets: Sequence[Set]) -> bool:
    """Checks if all sets in a list are pairwise disjoint (have no common elements).

    Args:
        sets: A list of sets.

    Returns:
        True if all sets are disjoint, False otherwise.
    """
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if not sets[i].isdisjoint(sets[j]):
                return False
    return True


def _iterate_nested(d):
    for key, value in d.items():
        yield key, value
        if isinstance(value, dict):
            yield from _iterate_nested(value)  # Recursively process the nested dict


def get_group_by_nested_attrs_func(
        sort_key_for_grouped_items_func, constant_keys_to_remove=set()):
    def ret(items: List[dict[K, Any]],
            attrs_groups_list: List[Set[K]],
            dict_to_str_func=dict_to_str,
            add_extra_index=-1):
        if not _are_all_disjoint(attrs_groups_list):
            raise ValueError("Attributes in the groups are not disjoint")
        keys = set().union(*[set(item.keys()) for item in items])

        for attrs in attrs_groups_list:
            if not attrs.issubset(keys):
                warnings.warn(
                    f"A group of attributes is not in the items: {attrs}")

        # Remove those that we don't have
        attrs_groups_list = [
            attrs & keys for attrs in attrs_groups_list
        ]

        vals_dict = {
            k: {item[k] for item in items if k in item}
            for k in keys
        }
        constant_keys = {k for k in keys if len(vals_dict[k]) == 1}
        constant_keys -= constant_keys_to_remove

        # print(f"{attrs_groups_list=}")
        # print(f"{constant_keys=}")

        ## TEMPORARY COMMENT THIS LINE OUT FOR INFORMS; TODO: DEBUG.
        ## PROBLEM: When there is only one NN in the "line" level, then it
        ## groups the NN with the PBGI GP method (this is what was observed)
        # attrs_groups_list = [z - constant_keys for z in attrs_groups_list]

        attrs_groups_list = [z for z in attrs_groups_list if len(z) != 0]

        # if len(attrs_groups_list) == 0:
        #     raise ValueError("No attributes to group by")

        ret = _group_by_nested_attrs(
            items, [set()] if len(attrs_groups_list) == 0 else attrs_groups_list,
            dict_to_str_func,
            sort_key_for_grouped_items_func=sort_key_for_grouped_items_func)

        ## At this point, this auto code is broken, I don't know how to fix, I've given up

        nonconstant_keys = set()
        keys_in_all = {u for u in keys}

        for key, value in _iterate_nested(ret):
            if not (key == "items" and isinstance(value, list)):
                continue
            itmz = value
            nonconstant_keys_item = set()
            in_all_keys_item = set()
            for k in keys:
                is_in_all = True
                vals_taken = set()
                for idx in itmz:
                    item = items[idx]
                    if k in item:
                        vals_taken.add(item[k])
                    else:
                        is_in_all = False
                if is_in_all:
                    in_all_keys_item.add(k)
                if len(vals_taken) > 1: # if non-constant
                    nonconstant_keys_item.add(k)
            nonconstant_keys |= nonconstant_keys_item
            keys_in_all &= in_all_keys_item

        nonconstant_keys -= {"index"}
        keys_in_all -= {"index"}

        nonconstant_keys_in_all = nonconstant_keys & keys_in_all
        nonconstant_keys_not_in_all = nonconstant_keys - nonconstant_keys_in_all

        # print(f"{nonconstant_keys_in_all=}, {nonconstant_keys_not_in_all=}")

        if len(nonconstant_keys_in_all) != 0:
            attrs_groups_list = [nonconstant_keys_in_all] + attrs_groups_list
            return ret(items, attrs_groups_list, dict_to_str_func,
                       add_extra_index=add_extra_index)

        if len(nonconstant_keys_not_in_all) != 0:
            attrs_groups_list[add_extra_index] |= nonconstant_keys_not_in_all

        return _group_by_nested_attrs(
            items, attrs_groups_list, dict_to_str_func,
            return_single=True,
            sort_key_for_grouped_items_func=sort_key_for_grouped_items_func), attrs_groups_list
    return ret
