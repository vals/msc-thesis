"""Toolbox for performing calculations with Grassmannians."""
import numpy as np
import sympy
import itertools

def grassmann_coordinates(A):
    """Returns the Grassmann coordinates of the span (column space) of a matrix A"""
    height, width = A.shape
    return np.array([np.linalg.det(A[np.ix_(comb)]) for comb
            in itertools.combinations(range(height), width)])

def parity(permutation):
    """Returns the parity (sign) of a permutation.
    Code is from StackOverflow, users Ashwin and Weeble.
    """
    permutation = list(permutation)
    length = len(permutation)
    elements_seen = [False] * length
    cycles = 0
    for index, already_seen in enumerate(elements_seen):
        if already_seen:
            continue
        cycles += 1
        current = index
        while not elements_seen[current]:
            elements_seen[current] = True
            current = permutation[current]
    return 1 - 2*((length-cycles) % 2)

def hodge_star_coeff(indices, dimension):
    """Return the coefficient of the dual basis multivector, it is either 1 or -1."""
    permutation = list(set(range(dimension)).difference(set(indices))) + list(indices)
    return parity(permutation)

# Hypothesis:
# If a list of combinations of k out of n numbers is given in lexicographic order,
# then when one takes the complement in each post of the list, the new list will be given
# in _reverse_ lexicographical order. (If the list posts are assumed sorted.)

def dual(grassmann_coordinates, dimension, subdimension):
    """Returns the dual Grassmann coordinates of given Grassmann coordinates."""
    signs = np.array([hodge_star_coeff(indices, dimension) for indices
            in itertools.combinations(range(dimension), subdimension)])
    # This works if the hypothesis is true
    return grassmann_coordinates[::-1]*signs

def wedge(grassmann_coordinates, vector, grade):
    """Returns Grassmann coordinates of the exterior product of a linear spaces
    represented by its Grassmann coordinates and a vector.
    """
    # Recall that a vector for all intents and purposes can be seen as the Grassmann
    # coordinates of a one-dimensional linar subspace; a line.

    # Generate dictionary for converting between index representations for the
    # grassmann_coordinates.
    index = {}
    coordinate = 0
    dimension = vector.shape[0]
    for comb in itertools.combinations(range(dimension), grade):
        index[comb] = coordinate
        coordinate += 1

    # Generate dictionary for index representations for the output coordinates
    wedge_index = {}
    coordinate = 0
    dimension = vector.shape[0]
    for comb in itertools.combinations(range(dimension), grade + 1):
        wedge_index[comb] = coordinate
        coordinate += 1

    # TODO: Make dictionary generating function, this is ugly copy-pasting.

    # Sum up products by index combinations
    wedge_product = np.zeros(sympy.binomial(dimension, grade + 1))
    for indices, listlocations in index.iteritems():
        for i in range(dimension):
            # We are adding to the LEX-first combination. So the sign is determined by whether
            # the current combination we are looking at is an even or odd permutation of the
            # LEX-first combination.
            # Idea: Make a map from LEX-first combination to the first natural numbers, apply map
            # to the current combination, and from the resulting we can calculate the sign using parity().
            combination_map = {}
            number = 0
            for j in tuple( set(indices).union( set((i,)) ) ):
                combination_map[j] = number
                number += 1
            permutation = [combination_map[j] for j in list(indices) + [i]]
            sign = parity(permutation)

            try:
                wedge_product[wedge_index[tuple( set(indices).union( set((i,)) ) )]
                ] += sign*grassmann_coordinates[index[indices]]*vector[i]
            except KeyError:
                # If the union gives a result not in wedge_index, the resulting product on
                # the right side equals zero.
                pass

    return wedge_product

def plucker_relations(n, k):
    """Returns an array of tuples, where every tuple is a pair of indexes of the coefficients that
    are multiplied with eachother in the plucker relation.
    NOTE: At the moment, only returns one relation, to get things working.
    """
    # This way of generating index sets only gives special cases, but it assures
    # everything is defined, when experimenting.
    I = range(k+1)
    J = range(n, n-k+1, -1)
    P = []
    for i in I:
        P += [(tuple(set(I).difference(set((i,)))), tuple(set(J).union(set((i,)))))]

    return P

def grid_points3(nx = 2., ny = 2., nz = 2.):
    """Returns the augmented coordinate matrix for the points in cubic grid (in R^3), where the number of points
    along each axis is given as the parameter.
    """
    grid = np.empty((0, 4))
    for x in np.arange(min(nx + 1., 0.), max(1., nx)):
        for y in np.arange(min(ny + 1., 0.), max(1., ny)):
            for z in np.arange(min(nz + 1., 0.), max(1., nz)):
                grid = np.vstack((grid, np.array([x, y, z, 1.0])))
    
    return grid.T