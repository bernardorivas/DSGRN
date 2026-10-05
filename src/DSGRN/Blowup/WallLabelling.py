### WallLabelling.py
### MIT LICENSE 2024 Marcio Gameiro

import pychomp
import numpy as np

def ramp_system_wall_labelling(gamma, theta, h, f_ramp, global_bound=None, legacy=False):
    """Compute wall labelling from a ramp system.

    Definition `defn:ramp-wall-labeling` (RampSystemsv4.tex) evaluates the
    right most walls at the sentinel threshold GB_n of eq:GAB, so they always
    point inward. GB_n is computed from the ramp system unless `global_bound`
    (a list of GB_n) is given. `legacy=True` uses theta + 10 * h instead, the
    previous surrogate, which depends on h and can label an outer wall as an
    exit, i.e. the labelling need not be strongly dissipative.
    """
    # Space dimension
    dim = len(gamma)
    # Get outgoing theta and h
    theta_out, h_out = [], []
    for n in range(dim):
        theta_n, h_n = [], []
        for k in range(dim):
            theta_n.extend(theta[k][n])
            h_n.extend(h[k][n])
        theta_out.append(theta_n)
        h_out.append(h_n)
    # Sort theta and h
    theta_s, h_s = [], []
    for n in range(dim):
        theta_n = theta_out[n]
        h_n = h_out[n]
        sorted_indices = np.argsort(theta_n)
        theta_n_sorted = [theta_n[k] for k in sorted_indices]
        h_n_sorted = [h_n[k] for k in sorted_indices]
        theta_s.append(theta_n_sorted)
        h_s.append(h_n_sorted)
    # Number of thresholds (thetas) in each dimension
    num_thetas = [len(theta_s[n]) for n in range(dim)]

    def cell_point(coords):
        """Return a point in the cell where the ramp functions are constant"""
        # Take theta - h, or theta + h for the right most cell. A node without
        # outgoing thresholds regulates nothing, so any value works there.
        return [theta_s[n][k] - h_s[n][k] if k < num_thetas[n]
                else (theta_s[n][k - 1] + h_s[n][k - 1] if k > 0 else 0)
                for n, k in enumerate(coords)]

    def cell_wall_label(top_cell):
        """Return wall labelling of a top cell"""
        # Get cell coordinates
        coords = cc.coordinates(top_cell)
        # Evaluate the ramp system at a point of the cell
        f_x_cell = f_cell[top_cell]
        # Get the theta values at the left walls of cell
        theta_left = [theta_s[n][k - 1] if k > 0 else 0 for n, k in enumerate(coords)]
        # Get the theta values at the right walls of cell (take GB_n at the right most
        # wall, or theta + 10 * h in legacy mode)
        theta_right = [theta_s[n][k] if k < num_thetas[n]
                       else (theta_s[n][k - 1] + 10 * h_s[n][k - 1] if legacy and k > 0
                             else global_bound[n])
                       for n, k in enumerate(coords)]
        # Get values of the ramp system on the left walls
        left_wall_vals = [-gamma[n] * theta_left[n] + f_x_cell[n] for n in range(dim)]
        # Get values of the ramp system on the right walls
        right_wall_vals = [-gamma[n] * theta_right[n] + f_x_cell[n] for n in range(dim)]
        # Make sure the values at the walls are nonzero
        if any(val == 0 for val in left_wall_vals):
            raise ValueError('Ramp system evaluates to zero on left wall.')
        if any(val == 0 for val in right_wall_vals):
            raise ValueError('Ramp system evaluates to zero on right wall.')
        # Get the signs (wall labelling) on the left walls
        left_wall_signs = [-1 if val < 0 else 1 for val in left_wall_vals]
        # Get the signs (wall labelling) on the right walls
        right_wall_signs = [-1 if val < 0 else 1 for val in right_wall_vals]
        # Get bit-wise representation of wall labelling
        wall_label = 0
        for n in range(dim):
            # Set left walls sign
            if left_wall_signs[n] == -1:
                # Set bit (move left)
                wall_label |= (1 << n)
            # Set right walls sign
            if right_wall_signs[n] == 1:
                # Set bit (move right)
                wall_label |= (1 << n + dim)
        return wall_label

    # Create cubical complex
    num_boxes = [k + 1 for k in num_thetas]
    cc = pychomp.CubicalComplex(num_boxes)
    # Values of the ramp nonlinearities, which are constant on each top cell
    f_cell = {top_cell: f_ramp(cell_point(cc.coordinates(top_cell))) for top_cell in cc(dim)}
    if global_bound is None:
        # GB_n = max{M^E_n / gamma_n, largest theta + its h} + 1 (eq:GAB), where
        # M^E_n is the maximum of the n-th ramp nonlinearity
        global_bound = []
        for n in range(dim):
            max_E = max(f_x[n] for f_x in f_cell.values())
            theta_h = theta_s[n][-1] + h_s[n][-1] if num_thetas[n] > 0 else 0
            global_bound.append(max(max_E / gamma[n], theta_h) + 1)
    # Get labelling of each top cell
    labelling = []
    for top_cell in cc(dim):
        wall_label = cell_wall_label(top_cell)
        labelling.append(wall_label)
    return labelling, num_thetas
