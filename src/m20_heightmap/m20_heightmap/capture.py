"""Measured map support, independent of the unchanged model output grid."""
import math
import numpy as np
from .contract import GRID, ROWS, COLS, RESOLUTION


def capture_geometry(padding_m):
    if (type(padding_m) not in (float, int) or not math.isfinite(padding_m)
            or not 0 <= padding_m <= 2):
        raise ValueError('map_capture_padding_m must be finite in [0,2]')
    cells = math.ceil(padding_m/RESOLUTION-1e-7)
    if cells == 0:
        return GRID.copy(), (ROWS, COLS), np.array([.85, .55])
    rows, cols = ROWS+2*cells, COLS+2*cells
    y, x = np.meshgrid((np.arange(rows)-rows//2)*RESOLUTION,
                       (np.arange(cols)-cols//2)*RESOLUTION, indexing='ij')
    return np.column_stack((x.ravel(), y.ravel())), (rows, cols), np.array([.85, .55])+cells*RESOLUTION


def capture_memory_budget(estimator):
    grid, shape, _ = capture_geometry(estimator.get('map_capture_padding_m', 0.))
    # Full supported patch: six float64 arrays and two Boolean arrays.
    # Fitting reserve covers sampled point matrices, robust-fit temporaries,
    # and worker-to-parent transfer. Raw-cloud/ROS/LIO budget is separate.
    packets = len(grid)*estimator['max_packets']*50
    fit = len(grid)*estimator['max_points_per_cell']*512
    return {'capture_shape':list(shape),'capture_points':len(grid),
            'capture_packet_bytes':packets,'capture_fit_reserve_bytes':fit,
            'capture_reserve_bytes':2*(packets+fit)}
