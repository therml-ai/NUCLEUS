# Physics-related functions

This directly holds functions for "physics"-stuff.

## Grids

A lot of the functions rely on finite-differences, and depend strongly
on the grid that is supplied. For example, grids can be cell-centered,
face-centered, or nodal. 

Temperature and the SDF should always be cell-centered. The velocity
should be face-centered. Some functions may return a nodal grid.

This can sometimes be tricky. For instance, the velocity divergence takes face-centered
grid as input, but outputs a cell-centered grid.

## Non-dimensionalized

These functions all assume that the inputs are non-dimensionalized w.r.t.
reference quantities and are "physical".
Some functions may not work if supplied with normalized functions for neural
network inputs.