"""Collect generators of typical indicator functions for the Markov chain"""

import numpy as _np

def ball(center = _np.zeros(3), radius = 1., bdy = True):
    '''Returns the indicator function of a ball.

    :param center:

        A vector-like numpy array, defining the center of the ball.\n
        len(center) fixes the dimension.

    :param radius:

        Float or int, the radius of the ball

    :param bdy:

        Bool, when bdy=True and x is at the ball's boundary then
        ball_indicator(x) returns True, otherwise ball_indicator(x)
        returns False.

    Using standard values, the indicator of the 3-dim unit ball
    (including the boundary) is returned.

    '''
    dim = len(center)

    def ball_indicator(x):
        if len(x) != dim:
            raise ValueError('input has wrong dimension (%i istead of %i)' %(len(x), dim))
        if _np.linalg.norm(x) < radius:
            return True
        if bdy and _np.linalg.norm(x) == radius:
            return True
        return False

    # write docstring for ball_indicator
    ball_indicator.__doc__  = 'automatically generated ball indicator function:'
    ball_indicator.__doc__ += '\ncenter = ' + repr(center)[6:-1]
    ball_indicator.__doc__ += '\nradius = ' + str(radius)
    ball_indicator.__doc__ += '\nbdy    = ' + str(bdy)

    return ball_indicator

def hyperrectangle(lower = _np.zeros(3), upper = _np.ones(3), bdy = True):
    '''Returns the indicator function of a hyperrectangle.

    :param lower:

        Vector-like numpy array, defining the lower boundary of the hyperrectangle.\n
        len(lower) fixes the dimension.

    :param upper:

        Vector-like numpy array, defining the upper boundary of the hyperrectangle.\n

    :param bdy:

        Bool, when bdy=True and x is at the hyperrectangles's boundary then
        hr_indicator(x) returns True, otherwise hr_indicator(x) returns False.

    Using standard values, the indicator of [0,1]**3 is returned.

    '''
    dim = len(lower)
    if (upper < lower).any():
        raise ValueError('invalid input; found upper < lower')

    if bdy:
        def hr_indicator(x):
            if len(x) != dim:
                raise ValueError('input has wrong dimension (%i istead of %i)' %(len(x), dim))
            if (lower <= x).all() and (x <= upper).all():
                return True
            return False
    else:
        def hr_indicator(x):
            if len(x) != dim:
                raise ValueError('input has wrong dimension (%i istead of %i)' %(len(x), dim))
            if (lower < x).all() and (x < upper).all():
                return True
            return False

    # write docstring for ball_indicator
    hr_indicator.__doc__  = 'automatically generated hyperrectangle indicator function:'
    hr_indicator.__doc__ += '\nlower = ' + repr(lower)[6:-1]
    hr_indicator.__doc__ += '\nupper = ' + repr(upper)[6:-1]
    hr_indicator.__doc__ += '\nbdy   = ' + str(bdy)

    return hr_indicator