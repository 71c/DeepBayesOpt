from utils_general.utils import dict_to_str


def _get_sort_key_for_param(k, v):
    """
    Generate a sort key for legend parameters that handles both parameter names
    and their values intelligently.

    Returns a tuple of (priority, param_name, numeric_value_for_sorting)
    """
    priority = 1

    if k == "attr_name":
        return (2, k, 0, v)

    # NN methods should come first (lowest priority values)
    if k == "nn.method":
        if v == "mse_ei":
            priority = 0.1
        else:
            priority = 0.2

    # GP methods should come after NN methods (medium priority values)
    if k == "gp_af":
        if v == "EI" or v == "LogEI":
            priority = 1.1
        else:
            priority = 1.2

    # "method" field is used for random search (highest priority when set)
    if k == "method":
        if v == "random search":
            priority = 2.0
        else:
            priority = 0.5

    # Strip "nn." prefix for cleaner display (except for nn.lamda)
    display_key = k
    if k != "nn.lamda" and k.startswith("nn."):
        display_key = k[3:]

    # Special handling for HPO-B search space IDs to include dimension
    if k == "objective.hpob_search_space_id" or k == "hpob_search_space_id":
        # Lazy import to avoid circular dependency
        from dataset.hpob_dataset import get_hpob_dataset_dimension
        dim = get_hpob_dataset_dimension(v)
        return (dim, display_key, 0, f"{display_key}={v} ({dim}D)")

    # Try to extract numeric value for proper sorting
    numeric_value = 0
    if isinstance(v, (int, float)):
        numeric_value = float(v)
    elif isinstance(v, str):
        try:
            # Try to parse as float (handles scientific notation like 5.2e-05)
            numeric_value = float(v)
        except (ValueError, TypeError):
            # Not a number, keep as 0 (will sort alphabetically by string)
            pass

    return (priority, display_key, numeric_value, f"{display_key}={v}")


def sort_key_for_grouped_items(item):
    """
    Create a sort key for grouped items based on their parameter values.
    Uses the same priority logic as get_sort_key_for_param to ensure
    consistent legend ordering (NN methods before GP methods before random search).
    """
    key, value_dict = item
    vals = value_dict['vals']

    # Determine the primary category (NN, GP, or random search) based on the parameters
    # This ensures the main grouping is correct before sorting by other parameters
    primary_priority = 1.0  # Default priority for other parameters

    # Check for method-identifying parameters
    if 'method' in vals:
        if vals['method'] == 'random search':
            primary_priority = 2.0  # Random search last
        else:
            primary_priority = 0.5  # Other methods
    elif 'gp_af' in vals:
        primary_priority = 1.1  # GP methods in the middle
    elif 'nn.method' in vals:
        primary_priority = 0.1 if vals['nn.method'] == 'mse_ei' else 0.2  # NN methods first
    elif any(k.startswith('nn.') for k in vals.keys()):
        # If there are NN parameters but no explicit method, assume it's an NN method
        primary_priority = 0.15  # Between mse_ei and other NN methods
    elif any(k.startswith('gp_af.') for k in vals.keys()):
        # If there are GP parameters but no explicit gp_af, assume it's a GP method
        primary_priority = 1.15

    # Create a sort key using the same logic as get_sort_key_for_param
    # Start with the primary priority to ensure main grouping
    sort_components = [(primary_priority,)]

    # Then add individual parameter sort keys
    for param_name in sorted(vals.keys()):
        param_value = vals[param_name]

        # Use the same sorting logic as plot_dict_to_str
        sort_key = _get_sort_key_for_param(param_name, param_value)
        # sort_key is (priority, display_key, numeric_value, formatted_string)
        # We use (priority, display_key, numeric_value) for sorting
        sort_components.append(sort_key[:-1])

    return tuple(sort_components)


def plot_key_value_to_str(k, v):
    """
    Convert a key-value pair to a string for plotting legend.
    Returns a tuple of (sort_key, formatted_string).
    """
    sort_key = _get_sort_key_for_param(k, v)
    # Return (sort_key[:-1], formatted_string) - exclude the formatted string from sort key
    return (sort_key[:-1], sort_key[-1])


def plot_dict_to_str(d):
    d_items = list(d.items())
    d_items.sort(key=lambda kv: plot_key_value_to_str(*kv))
    for key_name, prefix, plot_name in [
        ("nn.method", "nn.", "NN, method="),
        ("method", "", "method="),
        ("gp_af", "gp_af.", "GP, "),
        ("transfer_bo_method", "", "")
    ]:
        if key_name in d:
            d_method = {}
            d_non_method = {}
            for k, v in d_items:
                if k == key_name:
                    continue
                if k.startswith(prefix):
                    d_method[k[len(prefix):]] = v
                else:
                    d_non_method[k] = v
            method = d[key_name]
            if method == "random search":
                ret = method
            else:
                ret = f"{plot_name}{method}"
            if d_method:
                s = dict_to_str(d_method, include_space=True)
                ret += f" ({s})"
            if d_non_method:
                s = dict_to_str(d_non_method, include_space=True)
                ret += f", {s}"
            return ret

    items = [
        plot_key_value_to_str(k, v)
        for k, v in d_items
    ]
    return ", ".join([str(item[1]) for item in items])
