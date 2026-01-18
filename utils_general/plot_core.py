"""Core declarative plotting utilities.

Provides a to_plot dictionary pattern for building plots declaratively,
following a consistent structure across projects.

This module is shared between pandoras_box_llm and DeepBayesOpt.
The default behavior matches DeepBayesOpt exactly.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba_array

# =============================================================================
# Color Constants (shared between both projects)
# =============================================================================

BLUE = '#1f77b4'    # Training data
ORANGE = '#ff7f0e'  # Validation/Test data
GREEN = '#2ca02c'   # Optimal/reference values
RED = '#d62728'     # Alternative reference values


# =============================================================================
# Core Rendering (matches DeepBayesOpt behavior exactly by default)
# =============================================================================

def render_to_plot(
    ax: Axes,
    to_plot: Dict[str, Any],
    x_values: np.ndarray,
    plot_name: Optional[str] = None,
    line_properties: Optional[List[str]] = None,
    grid_alpha: Optional[float] = None,
):
    """Render a to_plot dictionary to a matplotlib axis.

    This function matches DeepBayesOpt's rendering behavior exactly by default.

    The to_plot dict structure (from DeepBayesOpt):
        - 'lines': List of line dicts with 'label', 'data', 'color', etc.
        - 'consts': List of horizontal reference line dicts
        - 'title': Plot title (required)
        - 'xlabel': X-axis label (required)
        - 'ylabel': Y-axis label (required)
        - 'log_scale_x': Whether to use log scale on x-axis (required)
        - 'log_scale_y': Whether to use log scale on y-axis (required)

    Args:
        ax: Matplotlib axes to render to
        to_plot: Dictionary with plot specification (see above)
        x_values: X-axis values (e.g., epoch numbers)
        plot_name: Optional prefix for title. If provided, title becomes
            "{plot_name} ({to_plot['title']})". Matches DeepBayesOpt behavior.
        line_properties: List of properties to extract from line dicts.
            Defaults to ['label', 'marker', 'linestyle', 'color'] (DeepBayesOpt default).
            pandoras_box_llm uses ['label', 'marker', 'linestyle', 'color', 'linewidth', 'alpha'].
        grid_alpha: Alpha for grid. None means no alpha (DeepBayesOpt default).
            pandoras_box_llm uses 0.3.
    """
    # Default line_properties matches DeepBayesOpt exactly
    if line_properties is None:
        line_properties = ['label', 'marker', 'linestyle', 'color']

    # Plot lines (DeepBayesOpt uses `if lines is not None:`)
    lines = to_plot.get('lines')
    if lines is not None:
        for line in lines:
            kwargs = {p: line[p] for p in line_properties if p in line}
            ax.plot(x_values, line['data'], **kwargs)

    # Plot constant reference lines
    consts = to_plot.get('consts')
    if consts is not None:
        for line in consts:
            kwargs = {p: line[p] for p in line_properties if p in line}
            ax.axhline(line['data'], **kwargs)

    # Title handling (matches DeepBayesOpt exactly)
    plot_desc = to_plot['title']
    if plot_name is not None:
        title = f"{plot_name} ({plot_desc})"
    else:
        title = plot_desc
    ax.set_title(title)

    # Axis labels (DeepBayesOpt uses direct access)
    ax.set_xlabel(to_plot['xlabel'])
    ax.set_ylabel(to_plot['ylabel'])

    ax.legend()

    # Grid (DeepBayesOpt uses ax.grid(True) without alpha)
    if grid_alpha is not None:
        ax.grid(True, alpha=grid_alpha)
    else:
        ax.grid(True)

    # Log scales (DeepBayesOpt uses direct access)
    if to_plot['log_scale_x']:
        ax.set_xscale('log')
    if to_plot['log_scale_y']:
        ax.set_yscale('log')


# =============================================================================
# Line Building Utilities
# =============================================================================

def build_train_test_lines(
    train_data: np.ndarray,
    test_data: np.ndarray,
    label: str = '',
    color=None,
    alpha: float = 1.0,
    train_label_base: str = 'Train (NN)',
    test_label_base: str = 'Test (NN)',
) -> List[Dict[str, Any]]:
    """Build lines list for train/test data (DeepBayesOpt style).

    Matches DeepBayesOpt's behavior in plot_acquisition_function_net_training_history_ax.

    Args:
        train_data: Training metric values
        test_data: Test metric values
        label: Optional label suffix for legend (e.g., model name)
        color: If provided, use same color for both lines (multi-model overlay mode)
        alpha: Base alpha value
        train_label_base: Base label for training line (default: 'Train (NN)')
        test_label_base: Base label for test line (default: 'Test (NN)')

    Returns:
        List of line dicts for use in to_plot['lines']
    """
    label_suffix = f' ({label})' if label else ''

    # Get colors using to_rgba_array (matches DeepBayesOpt)
    train_color = to_rgba_array(BLUE if color is None else color, alpha=alpha)[0]
    test_color = to_rgba_array(ORANGE if color is None else color, alpha=alpha)[0]

    lines = [
        {
            'label': f'{train_label_base}{label_suffix}',
            'data': train_data,
            'color': train_color
        },
        {
            'label': f'{test_label_base}{label_suffix}',
            'data': test_data,
            'color': test_color
        },
    ]

    if color is not None:
        # Multi-model overlay mode: make train line dashed and reduce alpha
        # Matches DeepBayesOpt lines 581-585
        lines[0]['linestyle'] = '--'
        lines[0]['color'][3] *= 0.7

    return lines


def build_const_line(
    value: float,
    label: str,
    color='k',
    linestyle: str = '--',
) -> Dict[str, Any]:
    """Build a horizontal reference line dict (DeepBayesOpt style).

    Matches DeepBayesOpt's const line format (lines 588-594).

    Args:
        value: Y-axis value for the horizontal line
        label: Legend label
        color: Line color (default: 'k' for black, matching DeepBayesOpt)
        linestyle: Line style (default: '--')

    Returns:
        Dict for use in to_plot['consts']
    """
    return {
        'label': label,
        'data': value,
        'color': color,
        'linestyle': linestyle,
    }


# =============================================================================
# Stats Extraction Utilities
# =============================================================================

def extract_train_test_from_stats_epochs(
    stats_epochs: List[Dict],
    stat_name: str,
    train_path: str = 'after_training',
) -> tuple:
    """Extract train/test metric arrays from stats_epochs (DeepBayesOpt style).

    Matches DeepBayesOpt's data extraction pattern (lines 530, 560-561).

    Args:
        stats_epochs: List of epoch stats dicts from training history
        stat_name: Stat name to extract (e.g., 'mse', 'gittins_loss')
        train_path: Path within 'train' dict (default: 'after_training')

    Returns:
        Tuple of (train_data, test_data) as numpy arrays
    """
    train_data = np.array(
        [epoch['train'][train_path][stat_name] for epoch in stats_epochs])
    test_data = np.array(
        [epoch['test'][stat_name] for epoch in stats_epochs])
    return train_data, test_data


# =============================================================================
# Aliases for pandoras_box_llm compatibility
# =============================================================================

# pandoras_box_llm uses 'validation' instead of 'test'
def extract_metric_from_stats_epochs(
    stats_epochs: List[Dict],
    metric_key: str,
    train_path: str = 'after_training',
    validation_key: str = 'validation',
    validation_fallback_key: str = 'test'
) -> tuple:
    """Extract train/validation metric arrays from stats_epochs.

    This version supports 'validation' key with fallback to 'test' for
    compatibility with pandoras_box_llm.

    Args:
        stats_epochs: List of epoch stats dicts from training history
        metric_key: Key to extract (e.g., 'loss', 'accuracy')
        train_path: Path within 'train' dict (default: 'after_training')
        validation_key: Key for validation data (default: 'validation')
        validation_fallback_key: Fallback key if validation_key not found (default: 'test')

    Returns:
        Tuple of (train_data, validation_data) as numpy arrays
    """
    train_data = []
    validation_data = []

    for epoch in stats_epochs:
        # Train data
        train_section = epoch.get('train', {}).get(train_path, {})
        train_val = train_section.get(metric_key)
        if isinstance(train_val, list) and len(train_val) == 1:
            train_val = train_val[0]
        train_data.append(train_val)

        # Validation data (with fallback)
        val_section = epoch.get(validation_key, epoch.get(validation_fallback_key, {}))
        val_val = val_section.get(metric_key)
        if isinstance(val_val, list) and len(val_val) == 1:
            val_val = val_val[0]
        validation_data.append(val_val)

    return np.array(train_data, dtype=float), np.array(validation_data, dtype=float)


def get_reference_value_from_stats_epochs(
    stats_epochs: List[Dict],
    key: str,
    validation_key: str = 'validation',
    validation_fallback_key: str = 'test'
) -> Optional[float]:
    """Get a reference value from the last epoch's validation stats.

    Args:
        stats_epochs: List of epoch stats dicts from training history
        key: Key for the reference value
        validation_key: Key for validation data (default: 'validation')
        validation_fallback_key: Fallback key if validation_key not found (default: 'test')

    Returns:
        Reference value or None if not available
    """
    if not stats_epochs:
        return None
    last_val = stats_epochs[-1].get(validation_key, stats_epochs[-1].get(validation_fallback_key, {}))
    return last_val.get(key)


# Alias for pandoras_box_llm naming convention
def build_train_validation_lines(
    train_data: np.ndarray,
    validation_data: np.ndarray,
    label: str = '',
    color=None,
    alpha: float = 1.0,
    linewidth: float = 2.0
) -> List[Dict[str, Any]]:
    """Build lines list for train/validation data (pandoras_box_llm style).

    This version includes 'linewidth' and 'alpha' in line dicts, matching
    pandoras_box_llm's original behavior.

    Args:
        train_data: Training metric values
        validation_data: Validation metric values
        label: Optional label suffix for legend (e.g., model name)
        color: If provided, use same color for both lines (multi-model overlay mode)
        alpha: Base alpha value
        linewidth: Line width

    Returns:
        List of line dicts for use in to_plot['lines']
    """
    label_suffix = f' ({label})' if label else ''

    if color is None:
        # Single-model mode: use distinct colors
        lines = [
            {
                'label': f'Train{label_suffix}',
                'data': train_data,
                'color': BLUE,
                'linewidth': linewidth,
                'alpha': alpha
            },
            {
                'label': f'Validation{label_suffix}',
                'data': validation_data,
                'color': ORANGE,
                'linewidth': linewidth,
                'alpha': alpha
            },
        ]
    else:
        # Multi-model overlay mode: same color, differentiate by linestyle
        train_rgba = to_rgba_array(color, alpha=alpha * 0.7)[0]
        val_rgba = to_rgba_array(color, alpha=alpha)[0]

        lines = [
            {
                'label': f'Train{label_suffix}',
                'data': train_data,
                'color': train_rgba,
                'linestyle': '--',
                'linewidth': linewidth
            },
            {
                'label': f'Validation{label_suffix}',
                'data': validation_data,
                'color': val_rgba,
                'linewidth': linewidth
            },
        ]

    return lines
