# v5 proof and scope

**Proven — v4 audit.** Put s=sqrt(1+q)-1. On real q in [0,Q], s<=Q/2,
s'<=1/2 and the positive Padé denominators are bounded below by their
values at zero. Applying product and quotient rules gives the v4 G'
majorant, with |exp(iK phi_P)|=1. Converting q derivatives to spatial
derivatives contributes 2*distance/z² and the field prefactor contributes
1/(lambda*z). The disk measure is dA/(25*pi), or 8*dA/(25*pi) on its octant.

**Proven — spatial differentiation.** q_x=2(x-x')/z²,
q_y=2(y-y')/z², q_xx=q_yy=2/z², q_xy=0.
U_i integrates G' q_i; U_ij integrates G'' q_i q_j+G' q_ij.
The aperture is fixed. All differentiated kernels are continuous on the
compact product of the closed aperture and a neighborhood of the ROI;
positive real denominators and the principal square-root branch provide
a uniform majorant. Dominated differentiation therefore applies.
Arb power series compute G derivatives of the exact stabilized Padé expression.

**Interval-certified — centers.** The area map uses eight signed/swapped
copies of x'=a*r*e(t), y'=a*r*t*e(t), with Jacobian a²*r*e(t)².
The same positive GL root/weight certificate as v3 is used. On each
complex parameter polydisk all branch and pole conditions are checked.
Cauchy and product-rule telescoping give 4*M*2^-(2n-1) times cell area
for each of the six integrands. Complex arithmetic is accumulated before
adding the error disks. Area GL is used here instead of boundary GL.

**Proven — local remainder and norm.** Along any unit segment in the disk,
|q'|<=2*distance/z², q''=2/z², q'''=0. Thus the third directional
derivative is bounded by area/(lambda*z) times
sup|G'''|*(2*distance/z²)^3+3*sup|G''|*(2*distance/z²)*(2/z²).
Real q interval subdivision certifies these suprema. The Taylor remainder
is <=M3*distance³/6; distance is bounded by radial plus outer-radius arc
distance. Polar cells cover the circular edge exactly. The quadratic
Taylor model is squared as a complex polynomial and integrated via exact
polar moments, including its signed cross terms. Add the local remainder
with Minkowski, combine disjoint weighted cell squares, then add the
certified Padé phase remainder norm. Arb encloses arithmetic throughout.

Largest-upper-contribution bisection is budgeted, not a guarantee of hitting
the requested tolerance. The geometric and continuous phase certificates
bound the same raw field norm. The oscillatory reference remains empirical.
