import numpy as np
import cellpylibhack.functions2d as cplhack

def evolveParallel(
    layer1, layer2, timesteps, apply_rule1, apply_rule2, neighbourhood="Moore"
):
    """Evolves two cellular automaton in parallel. Generates new generation for first layer then uses the output to generate 2nd layer."""
    assert (
        layer1.shape == layer2.shape,
        "Cannot evolve different layer shapes in parallel.",
    )

    r = 1  # only give vision of 3x3 centered around player - otherwise breaks movement

    von_neumann_mask = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
    for i in range(len(von_neumann_mask)):
        mask_size = np.absolute(r - i)
        von_neumann_mask[i][:mask_size] = 1
        if mask_size != 0:
            von_neumann_mask[i][-mask_size:] = 1

    _, rows, cols = layer1.shape
    neighbourhood_indices = cplhack._get_neighbourhood_indices(rows, cols, r)

    cell_indices = cplhack._get_cell_indices((rows, cols))
    cell_idx_to_neigh_idx = cplhack._get_cell_indices_to_neighbourhood_indices(
        cell_indices, rows, cols, r
    )

    # NOTE: to simplify total copied code, can only run this as fixed number of epochs.
    # TODO: add dynamic evolution.
    return _evolveParallel_fixed(
        layer1,
        layer2,
        timesteps,
        apply_rule1,
        apply_rule2,
        neighbourhood,
        rows,
        cols,
        neighbourhood_indices,
        von_neumann_mask,
        cell_indices,
        cell_idx_to_neigh_idx,
        r,
    )

def _evolveParallel_fixed(
    layer1,
    layer2,
    timesteps,
    apply_rule1,
    apply_rule2,
    neighbourhood,
    rows,
    cols,
    neighbourhood_indices,
    von_neumann_mask,
    cell_indices,
    cell_idx_to_neigh_idx,
    r,
):
    """
    Evolves the 2 layers of cellular automaton in parallel for a fixed of timesteps.


    """
    initial_conditions1 = layer1[-1]
    array1 = np.zeros((timesteps, rows, cols), dtype=layer1.dtype)
    array1[0] = initial_conditions1

    initial_conditions2 = layer2[-1]
    array2 = np.zeros((timesteps, rows, cols), dtype=layer2.dtype)
    array2[0] = initial_conditions2

    for t in range(1, timesteps):
        cell_layer1 = array1[t - 1]
        cell_layer2 = array2[t - 1]

        # get coords of all agents in matrix
        agent_neighbours = np.empty((0, 2), dtype=int)
        for (i, j), cell in np.ndenumerate(array1[t]):
            if cell > 1:  # if healthy or infected agent
                agent_neighbours = np.append(agent_neighbours, [[i, j]], axis=0)

        for row, cell_row1 in enumerate(cell_layer1):
            cell_row2 = cell_layer2[row]

            for col, cell1 in enumerate(cell_row1):
                cell2 = cell_row2[col]

                n1 = cplhack._get_neighbourhood(
                    cell_layer1,
                    neighbourhood_indices,
                    row,
                    col,
                    neighbourhood,
                    von_neumann_mask,
                )

                n2 = cplhack._get_neighbourhood(
                    cell_layer2,
                    neighbourhood_indices,
                    row,
                    col,
                    neighbourhood,
                    von_neumann_mask,
                )

                # NOTE: allow rule to return set of coordinates and new state.
                # This allows agents to move in environment by triggering rule from current position then setting next self position in different place - movement.

                # array1 is agent layer, so needs infection layer neighbours for agent location
                coords, value = apply_rule1(
                    n1, n2, (row, col), t, array1[t], array2[t], agent_neighbours
                )

                if value:  # if non-None value
                    x, y = coords
                    array1[t][x][y] = value

                # array2 is infection layer, so doesn't need infection layer neighbours for agent location, nor preview of next layer
                coords, value = apply_rule2(n2, (row, col), t, r)

                if value:  # if non-None value
                    x, y = coords
                    # add to new position, allow multiple cells to merge into a single point
                    array2[t][x][y] += value

    return np.concatenate((layer1, array1[1:]), axis=0), np.concatenate(
        (layer2, array2[1:]), axis=0
    )
