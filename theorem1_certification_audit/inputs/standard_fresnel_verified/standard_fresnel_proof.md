# Standard Fresnel extension

**Proven — physical definitions.** With unit illumination on the same
superellipse, let q=|r-s|²/z², k=2π/λ, K=kz and t=sqrt(1+q).
All three fields are exp(iK)/(iλz) times the aperture integral of their
carrier-stripped kernel. The common carrier is omitted for computation.
The exact scalar RS-I kernel is A exp(iK phi), with
A=1/(1+q)+i/(K t³), phi=q/(1+t). The projected spherical-spreading
factor in this convention is 1/(1+q). Padé retains the real amplitude
1/(1+q) and replaces the imaginary term by
i(1+q/4-q²/48)/(K(1+7q/4+35q²/48)), and replaces phi by
(q/2+q²/4)/(1+3q/4+q²/16).
Conventional Fresnel uses amplitude 1 and phase q/2.
Each error is an absolute raw complex field norm with uniform continuous
ROI measure dA/(25π), no phase fit or division by the RS-I norm.

**Proven — stable differences.** For either approximate amplitude B and
phase psi, D=exp(iK psi)[(A-B)+A expm1(iK(phi-psi))].
In particular, for Fresnel, A-1=-q/(1+q)+i/(K t³) and
phi-q/2=-q²/[2(1+t)²], by rationalizing t-1.
For Padé, the shared real amplitude cancels algebraically and the
remaining subtractions are formed in high precision balls. Arb encloses
any loss of precision. The v6 exact Padé formula is used only indirectly
through the archived result for a regression, not for calculating D.

**Interval-certified — approximation and norm.** The prior moment and
Gram implementations are reused, at AR=1. The dyadic domain enclosing
all physical q values comes from the largest source radius plus ROI radius.
On the whole rectangle enclosing the rho=2 ellipse, the code verifies
Re(1+q)>0 and, for Padé, excludes both rational poles by positive real
denominator enclosures. This verifies holomorphy through the ellipse.
Closed interval arcs cover its boundary; Arb encloses the maximum modulus
there and the maximum modulus principle covers the interior. The
degree-n Chebyshev–Lobatto interpolant error is 4 M rho^-n/(rho-1)
(Trefethen, ATAP, theorem 8.2), valid for complex values by the usual
aliasing argument with absolute values. This is the interpolation bound,
not the truncation bound.
All interpolant coefficients, beta/gamma source moments and Gram
operations are enclosed. Real/imaginary squares retain signed cross
terms, and their exact disk integral gives the continuous L2 squared.
With epsilon the uniform kernel error, triangle and reverse triangle
inequalities widen the polynomial norm by 100π epsilon/(λz), clipped
at zero on the lower side. These are two-sided enclosures.

**Empirical.** Direct 40-digit polar quadrature uses the three separate
kernels at finite orders in a circular case. Precision does not certify
spatial truncation. Timing and process high-water RSS are measurements,
not mathematical speedup statements. Novelty against the literature and
performance against a matched certifier remain unverified.

Theorem reference: https://people.maths.ox.ac.uk/trefethen/trefethen_sample.pdf
