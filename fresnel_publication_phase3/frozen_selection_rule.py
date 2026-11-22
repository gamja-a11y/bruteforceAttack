"""Phase 2 pilot-informed rule. All tolerance decisions use exact fractions."""
from fractions import Fraction


def start(params):
    if int(params["nf"]) >= 10:
        return 64, 512
    degree = 24 if params["model"] == "Fresnel" and params["nf"] == "1" and params["ar"] == "1" else 48
    return degree, 256


def advance(n, bits, lower, gram_width, delta, M, nf, absolute, relative):
    # Width in squared norm is compared to a squared field scale.
    width_goal = min(absolute, relative * lower) if lower > 0 else absolute
    if bits < 512 and gram_width > width_goal * max(lower, width_goal) and delta <= width_goal:
        return n, 512
    # rho=2. At fixed geometry the contour bound does not depend on n.
    # 2 * field slack = 8*pi*NF*M/2**n. Use pi < 22/7 for scheduling only.
    for degree in (32, 48, 64):
        if degree > n and (degree == 64 or 8*Fraction(22,7)*nf*M <= width_goal*2**degree):
            return degree, (512 if degree == 64 else bits)
    if bits < 512:
        return n, 512
    return None
