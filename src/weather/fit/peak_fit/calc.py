from weather.fit.format_check import is_x_formatted_correcly


Q = 0.01  # adjustment for the quadratic function
A = 1  # adjustment for the absolute value function

def calc_peak_profile(x, b, r):
    assert is_x_formatted_correcly(x)

    quadratic_term = -(r * Q * (x**2))
    abs_val_term = -abs((1 - r) * A * x)

    return quadratic_term + abs_val_term + b


def peak_temp_prepare(b: float):
    """
    r: roundness, k=1 is max roundness [0.1, 1] \n
    b: max temperature ~ y-intercept of sorts
    x: numpy array that is symmetric about 0
    """

    def peak_temp_fit(x, r):
        # assert k > 0 and k <= 1
        assert is_x_formatted_correcly(x)

        quadratic_term = -(r * Q * (x**2))
        abs_val_term = -abs((1 - r) * A * x)

        return quadratic_term + abs_val_term + b

    return peak_temp_fit
