"""Benchmark area, boundary, and piecewise-Chebyshev residual integration."""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import dataclass
from itertools import product
from math import comb, factorial
from pathlib import Path
from statistics import median
from time import perf_counter, process_time

import numpy as np
import scipy
from numpy.polynomial.chebyshev import chebint, chebval
from scipy.fft import dct

from fresnel_error_bound_validation import (
    AREA, ROI, Case, geometry, half_axes, mu, roi_rule, symbolic_proof,
)
from fresnel_quadrature_validation import unit_legendre, weighted_norm, write_csv


ORDERS = (32, 48, 64, 96)
DEGREES = (8, 12, 16, 24, 32)
SCALES = (1, 16, 256, 1024)
FIELD_TOL = 1e-4
BATCH = 8


def symbolic_checks():
    import sympy as s

    q = s.symbols("q", positive=True)
    h = s.Function("H")(q)
    g = s.Function("G")
    u = s.Dummy("u")
    h = s.Integral(g(u), (u, 0, q))/(2*q)
    assert s.simplify(2*h+2*q*s.diff(h, q)-g(q)) == 0
    return {
        "kernel_identity": symbolic_proof()["range"],
        "divergence_identity": "div((r'-r)H)=2H+2qH'=G. At q=0, H=G(0)/2.",
        "taylor_integral_remainder":
            "|exp(ix)-sum_j (ix)^j/j!|<=|x|^(m+1)/(m+1)! for real x, "
            "from the integral remainder and |exp(itx)|=1.",
        "moment_remainder_majorant":
            "rho <= sqrt(1+1/K^2) (K/512)^(m+1) J_[5(m+1)]/[(m+1)! lambda z].",
        "uniform_approximation_implication":
            "If sup|G-P|<=eps_G, area integration and normalized ROI weights give "
            "||L(G)-L(P)||_w <= A eps_G/(lambda z). A separate bound is required "
            "for boundary quadrature, the ROI norm, and floating-point error.",
    }


def phase_terms(q, k):
    t = np.sqrt(1+q)
    phi = q/(t+1)
    phi_p = (q/2+q*q/4)/(1+3*q/4+q*q/16)
    a = 1/(1+q)+1j/(k*t**3)
    delta = phi**5/(t**4+10*t*t+5)
    da = 1j/k*phi**5*(t*t+5*t+1)/(t**3*(35*t**4+14*t*t-1))
    return np.exp(1j*k*phi_p), a, da, 1j*k*delta


def residuals(q, k):
    phase, a, da, x = phase_terms(q, k)
    first = phase*(da+a*x)
    return np.stack((first, first+phase*a*x*x/2))


def q_max(case):
    a, b = sorted(half_axes(case.p,case.ar), reverse=True)
    if case.p>2:
        power = 2*case.p/(case.p-2)
        a *= (1+(b/a)**power)**(1/power)
    return float(((a+ROI)/case.z)**2)


def boundary_geometry(case, order):
    t,w = unit_legendre(order)
    phi = (np.arange(8)[:,None]+t)*np.pi/4
    phi = phi.ravel()
    w = np.tile(w*np.pi/4,8)
    ax,ay = half_axes(case.p,case.ar)
    c,s = np.cos(phi),np.sin(phi)
    # Use polar angle in scaled coordinates. The physical angle is
    # atan2(ay*sin(phi), ax*cos(phi)), with positive derivative. This puts
    # the sector breaks at the scaled diagonals even for a long aperture.
    f = abs(c)**case.p+abs(s)**case.p
    fp = case.p*(-s*np.sign(c)*abs(c)**(case.p-1)
                 +c*np.sign(s)*abs(s)**(case.p-1))
    r = f**(-1/case.p)
    rp = -r*fp/(case.p*f)
    x,y = ax*r*c,ay*r*s
    dx,dy = ax*(rp*c-r*s),ay*(rp*s+r*c)
    geom = (x,y,dy*w,-dx*w)
    constant_area = float(np.sum(x*geom[2]+y*geom[3])/2)
    return geom, abs(constant_area/AREA-1)


def h_direct(q,k,order=64):
    u,w = unit_legendre(order)
    # The substitution u=q*t removes the division by q exactly.
    return .5*np.sum(residuals(q[...,None]*u,k)*w,axis=-1)


@dataclass
class Primitive:
    edges: np.ndarray
    coefficients: list
    prefix: np.ndarray
    k: float
    validation_max: float
    fit_wall: float
    fit_cpu: float
    primitive_wall: float
    primitive_cpu: float

    def __call__(self,q):
        if np.any(q<0) or np.any(q>self.edges[-1]*(1+1e-12)):
            raise ValueError("Boundary q is outside the Chebyshev domain")
        result = np.empty((2,*q.shape),complex)
        for index,c in enumerate(self.coefficients):
            a,b = self.edges[index:index+2]
            mask = (q>=a)&(q<b if index+1<len(self.coefficients) else q<=b*(1+1e-12))
            vals = q[mask]
            if vals.size:
                z = 2*(vals-a)/(b-a)-1
                small = vals <= self.edges[1]*1e-4
                for m in range(2):
                    dst = np.zeros(vals.size,complex)
                    nonzero = ~small
                    dst[nonzero] = (self.prefix[index,m]+chebval(z[nonzero],c[m]))/(2*vals[nonzero])
                    result[m,mask] = dst
                if np.any(small):
                    direct = h_direct(vals[small],self.k,16)
                    for m in range(2):
                        result[m,mask] = np.where(small,0,result[m,mask])
                    indices = np.flatnonzero(mask)[small]
                    result.reshape(2,-1)[:,indices] = direct
        return result


def make_primitive(case,degree):
    start,cpu = perf_counter(),process_time()
    maximum = q_max(case)*(1+1e-13)
    scale = max(float(np.max(abs(residuals(np.linspace(0,maximum,101),case.k)))),1e-300)
    tolerance = scale*2e-10
    m = degree+1
    nodes = np.cos(np.pi*(np.arange(m)+.5)/m)
    test = np.linspace(-1,1,4*degree+3)
    pieces = []

    def split(a,b,depth):
        mapped = lambda t: (a+b)/2+(b-a)/2*t
        sample = residuals(mapped(nodes),case.k)
        coefficients = dct(sample,type=2,axis=-1)/m
        coefficients[:,0] *= .5
        truth = residuals(mapped(test),case.k)
        estimate = np.array([chebval(test,c) for c in coefficients])
        error = float(np.max(abs(estimate-truth)))
        if error>tolerance and depth<14:
            split(a,(a+b)/2,depth+1)
            split((a+b)/2,b,depth+1)
        else:
            pieces.append((a,b,coefficients,error))

    split(0.,maximum,0)
    fit_wall,fit_cpu = perf_counter()-start,process_time()-cpu
    start,cpu = perf_counter(),process_time()
    antiderivatives = []
    prefix = np.zeros((len(pieces)+1,2),complex)
    for index,(a,b,coeffs,_) in enumerate(pieces):
        anti = chebint(coeffs,axis=-1)*(b-a)/2
        anti[:,0] -= np.array([chebval(-1,c) for c in anti])
        prefix[index+1] = prefix[index]+np.array([chebval(1,c) for c in anti])
        antiderivatives.append(anti)
    return Primitive(np.array([part[0] for part in pieces]+[maximum]),antiderivatives,prefix,case.k,
                     max(part[3] for part in pieces),fit_wall,fit_cpu,
                     perf_counter()-start,process_time()-cpu)


def area_field(case,geom,x,y,part="all"):
    xa,ya,w = geom
    out = np.empty((5 if part=="all" else 2,x.size),complex)
    for index in range(0,x.size,BATCH):
        sl = slice(index,index+BATCH)
        q = ((x[sl,None]-xa)**2+(y[sl,None]-ya)**2)/case.z**2
        if part=="L":
            out[:,sl] = np.sum(residuals(q,case.k)*w,axis=-1)*case.pref
            continue
        e,a,da,t = phase_terms(q,case.k)
        r1 = e*(da+a*t)
        r2 = r1+e*a*t*t/2
        out[:3,sl] = np.array([np.sum(g*w,axis=1) for g in (r1,r2,e*(da+a*np.expm1(t)))])*case.pref
        for m in (1,2):
            out[2+m,sl] = np.sum(abs(a)*abs(t)**(m+1)*w,axis=1)*abs(case.pref)/factorial(m+1)
    return out


def rho_direct(case,geom,x,y):
    xa,ya,w = geom
    result = np.empty((2,x.size))
    for index in range(0,x.size,BATCH):
        sl = slice(index,index+BATCH)
        q = ((x[sl,None]-xa)**2+(y[sl,None]-ya)**2)/case.z**2
        t = np.sqrt(1+q)
        phase = case.k*(q/(t+1))**5/(t**4+10*t*t+5)
        mag = np.hypot(1/(1+q),1/(case.k*t**3))
        for m in (1,2):
            result[m-1,sl] = np.sum(mag*phase**(m+1)*w,axis=1)*abs(case.pref)/factorial(m+1)
    return result


def boundary_field(case,geom,x,y,primitive):
    xb,yb,nx,ny = geom
    out = np.empty((2,x.size),complex)
    batch = 8 if primitive is None else 48
    for index in range(0,x.size,batch):
        sl = slice(index,index+batch)
        dx,dy = xb-x[sl,None],yb-y[sl,None]
        q = (dx*dx+dy*dy)/case.z**2
        h = h_direct(q,case.k) if primitive is None else primitive(q)
        out[:,sl] = np.sum(h*(dx*nx+dy*ny),axis=-1)*case.pref
    return out


def moment_coefficients(case):
    ax,ay = half_axes(case.p,case.ar)
    moments = {(i,j):mu(i,j,case.p,ax,ay) for i in range(16) for j in range(16-i)}
    polys = []
    for n in (10,15):
        coefficients = np.zeros((n+1,n+1))
        for k in range(n+1):
            for i in range(k+1):
                for j in range(n-k+1):
                    coefficients[k-i,n-k-j] += (comb(n,k)*comb(2*k,2*i)*comb(2*(n-k),2*j)
                                                *moments[i,j]/case.z**(2*n))
        polys.append(coefficients)
    return polys


def moment_values(polys,x,y):
    return np.array([np.polynomial.polynomial.polyval2d(x*x,y*y,c) for c in polys])


def rho_from_moments(case,polys,x,y):
    j = moment_values(polys,x,y)
    scales = [abs(case.pref)*np.hypot(1,1/case.k)*(case.k/512)**(m+1)/factorial(m+1)
              for m in (1,2)]
    return j*np.array(scales)[:,None]


def moments_high_precision(case,polys):
    import mpmath as mp

    with mp.workdps(55):
        p,ar,nf,theta = [mp.mpf(str(v)) for v in (case.p,case.ar,case.nf,case.theta)]
        z = 10/theta
        ay = mp.sqrt(100*mp.pi*mp.gamma(1+2/p)/(4*ar*mp.gamma(1+1/p)**2))
        ax = ar*ay
        x,y = mp.mpf(3),mp.mpf(4)
        errors = []
        return_values = moment_values(polys,np.array([3.]),np.array([4.]))[:,0]
        for n,approx in zip((10,15),return_values):
            terms = []
            for k in range(n+1):
                for i in range(k+1):
                    for j in range(n-k+1):
                        moment = (4*ax**(2*i+1)*ay**(2*j+1)/p**2*mp.gamma((2*i+1)/p)
                                  *mp.gamma((2*j+1)/p)/mp.gamma(1+(2*i+2*j+2)/p))
                        terms.append(comb(n,k)*comb(2*k,2*i)*comb(2*(n-k),2*j)
                                     *x**(2*(k-i))*y**(2*(n-k-j))*moment)
            exact = mp.fsum(terms)/z**(2*n)
            errors.append(float(abs(mp.mpf(float(approx))/exact-1)))
    return errors


def metric(candidate,target,w):
    delta = candidate-target
    n = weighted_norm(target,w)
    phase_weight = w*abs(target)**2
    return {"complex_L1":float(np.dot(w,abs(candidate))),"complex_L2":weighted_norm(candidate,w),
            "relative_complex_L2":weighted_norm(delta,w)/max(n,1e-300),
            "max_complex_abs_error":float(np.max(abs(delta))),
            "phase_rms_rad":float(np.sqrt(np.dot(phase_weight,np.angle(candidate*target.conj())**2))
                                  /max(n,1e-300))}


def timed(fn,repeats=1):
    wall,cpu,result = [],[],None
    for _ in range(repeats):
        t,c = perf_counter(),process_time()
        result = fn()
        wall.append(perf_counter()-t)
        cpu.append(process_time()-c)
    return result,median(wall),median(cpu)


def audit(case,out,index):
    x,y,w = roi_rule(32,case.ar)
    tag = {**case.metadata(),"case":index}
    area_results, geoms, area_timing = {},{},{}
    for order in ORDERS:
        geoms[order],gw,gc = timed(lambda:geometry(case.p,case.ar,order))
        area_results[order],fw,fc = timed(lambda:area_field(case,geoms[order],x,y))
        area_timing[order] = (gw,gc,fw,fc)
    ref = area_results[96]
    actual = weighted_norm(ref[2],w)
    osc = [weighted_norm(ref[m],w)+weighted_norm(ref[m+3],w) for m in (0,1)]
    baseline_changes = {
        a:max(metric(area_results[a][m],ref[m],w)["relative_complex_L2"] for m in (0,1))
        for a in ORDERS
    }
    rho_refinements = {
        a: [weighted_norm(area_results[a][m+3]-ref[m+3],w)/max(weighted_norm(ref[m+3],w),1e-300)
            for m in (0,1)] for a in ORDERS
    }
    area_order = next(a for a in ORDERS if max(baseline_changes[a],*rho_refinements[a])<=FIELD_TOL)
    bound_ref_delta = [weighted_norm(area_results[64][m]-ref[m],w)
                       +weighted_norm(area_results[64][m+3]-ref[m+3],w) for m in (0,1)]
    rule = {}
    boundary_rows = []
    for order in ORDERS:
        (geom,area_err),gw,gc = timed(lambda:boundary_geometry(case,order))
        if area_err>1e-7:
            raise RuntimeError("Boundary orientation or Jacobian failed the G=1 area check")
        vals,wall,cpu = timed(lambda:boundary_field(case,geom,x,y,None))
        rule[order] = (geom,vals)
        for m in (0,1):
            boundary_rows.append({**tag,"boundary_order":order,"m":m+1,"G1_area_relative_error":area_err,
                                  **metric(vals[m],ref[m],w),"boundary_setup_wall_s":gw,
                                  "boundary_setup_cpu_s":gc,"evaluation_wall_s":wall,"evaluation_cpu_s":cpu})
    high_primitive,_,_ = timed(lambda:make_primitive(case,32))
    high_probe = boundary_field(case,rule[96][0],x,y,high_primitive)
    endpoint = high_primitive(np.array([0.]))
    np.testing.assert_array_equal(endpoint,0)
    chosen_order = next((a for a in ORDERS if max(
        metric(rule[a][1][m],ref[m],w)["relative_complex_L2"] for m in (0,1))<=FIELD_TOL), ORDERS[-1])
    chosen_geom = rule[chosen_order][0]
    polys,pw,pc = timed(lambda:moment_coefficients(case))
    rhos = rho_from_moments(case,polys,x,y)
    roi16_x,roi16_y,roi16_w = roi_rule(16,case.ar)
    roi16_field = boundary_field(case,rule[96][0],roi16_x,roi16_y,high_primitive)
    roi16_rho = rho_from_moments(case,polys,roi16_x,roi16_y)
    cheb_rows,budget,accuracy = [],[],[]
    primitives = {}
    for order in ORDERS:
        gw,gc,fw,fc = area_timing[order]
        vals = area_results[order]
        for m in (0,1):
            accuracy.append({**tag,"method":"A_area","order":order,"m":m+1,
                             **metric(vals[m],ref[m],w),"setup_wall_s":gw,"setup_cpu_s":gc,
                             "evaluation_wall_s":fw,"evaluation_cpu_s":fc,
                             "actual_RS_minus_PP":actual,"B0_osc":weighted_norm(vals[m],w)+weighted_norm(vals[m+3],w),
                             "B0_osc_over_actual":(weighted_norm(vals[m],w)+weighted_norm(vals[m+3],w))/actual,
                             "B0_osc_satisfied":weighted_norm(vals[m],w)+weighted_norm(vals[m+3],w)>=actual,
                             "rho_relative_L2_change_vs_96":rho_refinements[order][m],
                             "converged":metric(vals[m],ref[m],w)["relative_complex_L2"]<=FIELD_TOL
                             and rho_refinements[order][m]<=FIELD_TOL})
            value = rule[order][1][m]
            estimate = weighted_norm(value,w)+weighted_norm(ref[m+3],w)
            accuracy.append({**tag,"method":"B_boundary_GL","order":order,"m":m+1,
                             **metric(value,ref[m],w),"actual_RS_minus_PP":actual,"B0_osc":estimate,
                             "B0_osc_over_actual":estimate/actual,"B0_osc_satisfied":estimate>=actual,
                             "converged":metric(value,ref[m],w)["relative_complex_L2"]<=FIELD_TOL})
    for degree in DEGREES:
        primitive = make_primitive(case,degree)
        value,wall,cpu = timed(lambda:boundary_field(case,chosen_geom,x,y,primitive))
        primitives[degree] = (primitive,value)
        pieces = len(primitive.coefficients)
        storage = primitive.edges.nbytes+primitive.prefix.nbytes+sum(a.nbytes for a in primitive.coefficients)
        for m in (0,1):
            # Independent sampled diagnostics, deliberately not a rigorous
            # bound on boundary quadrature, interpolation, or ROI integration.
            cheb_gap = weighted_norm(value[m]-rule[chosen_order][1][m],w)
            boundary_gap = weighted_norm(rule[chosen_order][1][m]-rule[96][1][m],w)
            inner_gap = weighted_norm(high_probe[m]-rule[96][1][m],w)
            roi_gap = (abs(weighted_norm(roi16_field[m],roi16_w)-weighted_norm(high_probe[m],w))
                       +abs(weighted_norm(roi16_rho[m],roi16_w)-weighted_norm(rhos[m],w)))
            epsilon = cheb_gap+boundary_gap+inner_gap+roi_gap+bound_ref_delta[m]
            estimate = weighted_norm(value[m],w)+weighted_norm(rhos[m],w)+epsilon
            row = {**tag,"method":"C_chebyshev","degree":degree,"order":chosen_order,"m":m+1,
                   **metric(value[m],ref[m],w),"pieces":pieces,"coefficient_storage_bytes":storage,
                   "sampled_G_max_error":primitive.validation_max,"fit_wall_s":primitive.fit_wall,
                   "fit_cpu_s":primitive.fit_cpu,"primitive_wall_s":primitive.primitive_wall,
                   "primitive_cpu_s":primitive.primitive_cpu,"evaluation_wall_s":wall,"evaluation_cpu_s":cpu,
                   "rho_direct_norm":weighted_norm(ref[m+3],w),"rho_moment_upper_norm":weighted_norm(rhos[m],w),
                   "B0_osc":osc[m],"B_fast_empirical":estimate,"actual_RS_minus_PP":actual,
                   "B_fast_over_actual":estimate/actual,"B_fast_satisfied_empirically":estimate>=actual,
                   "B_fast_over_B0_osc2":estimate/osc[1],"within_1p01":estimate<=1.01*osc[1],
                   "within_1p001":estimate<=1.001*osc[1],
                   "converged":metric(value[m],ref[m],w)["relative_complex_L2"]<=FIELD_TOL,
                   "certified":False}
            accuracy.append(row)
            cheb_rows.append(row)
            budget.append({**tag,"m":m+1,"degree":degree,"certified":False,
                           "status":"empirical_only","epsilon_L_certified":None,
                           "epsilon_budget_empirical":epsilon,"chebyshev_primitive_gap":cheb_gap,
                           "boundary_order_gap":boundary_gap,"inner_rule_check":inner_gap,
                           "ROI_order_16_to_32_norm_gap":roi_gap,
                           "reference_area_order_64_to_96_error":bound_ref_delta[m],
                           "direct_rho_norm":weighted_norm(ref[m+3],w),
                           "analytic_moment_rho_discrete_norm":weighted_norm(rhos[m],w),
                           "B_fast_empirical":estimate})
    valid = [row for row in cheb_rows if row["m"]==2 and row["converged"]
             and row["B_fast_satisfied_empirically"] and row["within_1p01"]]
    chosen = min(valid,key=lambda r:r["evaluation_wall_s"]+r["fit_wall_s"]+r["primitive_wall_s"]) if valid else None
    np.savez_compressed(out/f"residual_fields_case_{index:03d}.npz", x=x,y=y,w=w,
                        **{f"A_order_{a}":v for a,v in area_results.items()},
                        **{f"B_order_{a}":v[1] for a,v in rule.items()},
                        **{f"C_degree_{d}":v[1] for d,v in primitives.items()})
    summary = {**tag,"actual":actual,"B0_osc":osc,
               "B0_conventional":float(case.constant*abs(case.pref)*weighted_norm(
                   _j5(case,x,y),w)),"reference_64_to_96_budget":bound_ref_delta,
               "reference_converged":max(baseline_changes[64],*rho_refinements[64])<=FIELD_TOL,
               "selected_degree":None if chosen is None else chosen["degree"],"area_order":area_order,
               "boundary_order":chosen_order,"moment_setup_wall_s":pw,"moment_setup_cpu_s":pc,
               "moment_55_digit_relative_errors":moments_high_precision(case,polys) if index==1 else None}
    return summary,accuracy,cheb_rows,boundary_rows,budget


def _j5(case,x,y):
    from fresnel_error_bound_validation import j5_moments
    return j5_moments(case,x,y)


def benchmark(case,summary):
    degree = summary["selected_degree"]
    if degree is None:
        return []
    rows = []
    for count in SCALES:
        if count == 1:
            radius,angle = ROI/np.sqrt(2),np.pi/8
            x,y,w = np.array([radius*np.cos(angle)]),np.array([radius*np.sin(angle)]),np.ones(1)
        else:
            x,y,w = roi_rule(int(np.sqrt(count)),case.ar)
        reference = area_field(case,geometry(case.p,case.ar,96),x,y)
        for method in ("A_area","B_boundary_GL","C_chebyshev"):
            make_geom = (lambda:geometry(case.p,case.ar,summary["area_order"])) if method=="A_area" else (
                lambda:boundary_geometry(case,summary["boundary_order"])[0])
            geom,geom_wall,geom_cpu = timed(make_geom,3)
            p = make_primitive(case,degree) if method=="C_chebyshev" else None
            polys,poly_wall,poly_cpu = timed(lambda:moment_coefficients(case),3) if method!="A_area" else (None,0.,0.)
            field_call = (lambda:area_field(case,geom,x,y,"L")) if method=="A_area" else (
                lambda:boundary_field(case,geom,x,y,p))
            rho_call = (lambda:rho_direct(case,geom,x,y)) if method=="A_area" else (
                lambda:rho_from_moments(case,polys,x,y))
            field,field_wall,field_cpu = timed(field_call,3)
            _,rho_wall,rho_cpu = timed(rho_call,3)
            fit_wall,fit_cpu = (p.fit_wall,p.fit_cpu) if p else (0.,0.)
            int_wall,int_cpu = (p.primitive_wall,p.primitive_cpu) if p else (0.,0.)
            warm_wall,warm_cpu = field_wall+rho_wall,field_cpu+rho_cpu
            total_wall = geom_wall+poly_wall+fit_wall+int_wall+warm_wall
            total_cpu = geom_cpu+poly_cpu+fit_cpu+int_cpu+warm_cpu
            rows.append({"points":count,"method":method,"repeat_count":3,"geometry_wall_s":geom_wall,
                         "geometry_cpu_s":geom_cpu,"moment_setup_wall_s":poly_wall,"moment_setup_cpu_s":poly_cpu,
                         "chebyshev_wall_s":fit_wall,"chebyshev_cpu_s":fit_cpu,"primitive_wall_s":int_wall,
                         "primitive_cpu_s":int_cpu,"field_wall_s":field_wall,"field_cpu_s":field_cpu,
                         "rho_wall_s":rho_wall,"rho_cpu_s":rho_cpu,"cold_total_wall_s":total_wall,
                         "cold_total_cpu_s":total_cpu,"warm_total_wall_s":warm_wall,"warm_total_cpu_s":warm_cpu,
                         "working_array_budget_bytes":int(128*len(geom[0])*(BATCH if method=="A_area"
                             else 8*64 if method=="B_boundary_GL" else 48)),
                         "m1_relative_L2":metric(field[0],reference[0],w)["relative_complex_L2"],
                         "m2_relative_L2":metric(field[1],reference[1],w)["relative_complex_L2"]})
        base = rows[-3]
        for row in rows[-3:]:
            row["same_field_accuracy_pass"] = max(
                row["m1_relative_L2"],row["m2_relative_L2"],
                base["m1_relative_L2"],base["m2_relative_L2"])<=FIELD_TOL
            row["cold_speedup"] = base["cold_total_wall_s"]/row["cold_total_wall_s"]
            row["warm_speedup"] = base["warm_total_wall_s"]/row["warm_total_wall_s"]
            for factor in (2,5,10):
                row[f"same_accuracy_speedup_ge_{factor}"] = (row["same_field_accuracy_pass"]
                                                            and row["cold_speedup"]>=factor)
    return rows


def plots(out,accuracy,bench):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in accuracy if r["case"]==1 and r["method"]=="C_chebyshev"]
    fig,ax = plt.subplots(figsize=(7,4),layout="constrained")
    for key,label in (("B_fast_empirical","Empirical fast estimate"),("B0_osc","2D reference"),
                      ("actual_RS_minus_PP","Observed residual")):
        ax.plot([f"{r['degree']}/{r['m']}" for r in rows],[r[key] for r in rows],"o-",label=label)
    ax.set(xlabel="Degree / Taylor order",ylabel="Weighted field norm")
    ax.legend()
    fig.savefig(out/"accuracy_comparison.png",dpi=150)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,4),layout="constrained")
    for m in (1,2):
        items = [r for r in rows if r["m"]==m]
        ax.loglog([r["evaluation_wall_s"]+r["fit_wall_s"]+r["primitive_wall_s"] for r in items],
                  [max(r["relative_complex_L2"],1e-17) for r in items],"o-",label=f"m={m}")
    ax.set(xlabel="Chebyshev setup + field time (s)",ylabel="Raw complex L2 relative error")
    ax.legend()
    fig.savefig(out/"accuracy_speed_tradeoff.png",dpi=150)
    plt.close(fig)
    if not bench:
        return
    methods = ("A_area","B_boundary_GL","C_chebyshev")
    fig,ax = plt.subplots(figsize=(7,4),layout="constrained")
    for method in methods:
        data = [r for r in bench if r["method"]==method]
        ax.loglog([r["points"] for r in data],[r["cold_total_wall_s"] for r in data],"o-",label=method)
    ax.set(xlabel="Observation points",ylabel="Setup + field + remainder time (s)")
    ax.legend()
    fig.savefig(out/"runtime_comparison.png",dpi=150)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(7,3.5),layout="constrained")
    labels = ("cold_speedup","warm_speedup")
    matrix = np.array([[r[key] for r in bench if r["method"]==method]
                      for method in methods[1:] for key in labels])
    img = ax.imshow(np.log10(matrix),aspect="auto")
    for i,j in np.ndindex(matrix.shape):
        ax.text(j,i,f"{matrix[i,j]:.1f}×",ha="center",va="center",color="white")
    ax.set(xticks=range(4),xticklabels=SCALES,yticks=range(4),
           yticklabels=["B cold","B warm","C cold","C warm"])
    fig.colorbar(img,ax=ax,label="log10(speedup)")
    fig.savefig(out/"speedup_heatmap.png",dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",action="store_true")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--case",type=float,nargs=4,metavar=("P","NF","THETA","AR"))
    parser.add_argument("--out",type=Path)
    args = parser.parse_args()
    primary = Case(12.,10.,.3)
    cases = [primary]
    if args.case:
        if args.all or args.quick or args.benchmark or min(args.case) <= 0 or args.case[0] < 1:
            parser.error("--case selects one physical condition and is used with --run or --plan")
        cases = [Case(*args.case)]
    elif args.all:
        cases = ([primary]+[Case(p,n,t) for p,n,t in product((2.,4.,8.,12.),(1.,3.,10.),(.05,.1,.2,.3))
                            if (p,n,t)!=(12.,10.,.3)]+[Case(12.,3.,.2,ar) for ar in (2.,4.)])
    elif args.run and not args.quick:
        cases += [Case(p,n,t,ar) for p,n,t,ar in ((2.,1.,.1,1.),(4.,3.,.2,1.),
                                                  (8.,10.,.3,1.),(12.,3.,.2,2.),(12.,3.,.2,4.))]
    pairs = len(cases)*1024*sum(8*a*a for a in ORDERS)
    boundary_nodes = len(cases)*1024*sum(8*a for a in ORDERS)
    print(f"{len(cases)} cases; area {pairs:,} sample pairs; boundary {boundary_nodes:,} pairs "
          "with 64 inner nodes before Chebyshev; batch array workspace ~80 MiB. "
          "Benchmark repeats are additional.",flush=True)
    if args.plan or not (args.quick or args.run or args.benchmark):
        return
    proof = symbolic_checks()
    label = ("fast_bound_case" if args.case else "fast_bound_all" if args.all else
             "fast_bound_run" if args.run and not args.quick else "fast_bound_quick")
    out = args.out or Path(__file__).resolve().parent/label
    try:
        out.mkdir(parents=True,exist_ok=False)
    except FileExistsError:
        parser.error("Choose a new --out directory; existing data are preserved.")
    results,accuracy,cheb,boundary,budget = [],[],[],[],[]
    for index,case in enumerate(cases,1):
        print(f"Case {index}/{len(cases)}: {case}",flush=True)
        result,a,c,b,e = audit(case,out,index)
        results.append(result)
        accuracy.extend(a)
        cheb.extend(c)
        boundary.extend(b)
        budget.extend(e)
        primary_pass = results[0]["reference_converged"] and results[0]["selected_degree"] is not None
        if index==1 and not primary_pass:
            print("WARNING: reference or empirical accuracy gate did not pass.",flush=True)
            break
    bench = benchmark(primary,results[0]) if args.benchmark and primary_pass else []
    for name,rows in (("fast_bound_accuracy",accuracy),("chebyshev_convergence",cheb),
                      ("boundary_quadrature_convergence",boundary),("certified_error_budget",budget)):
        write_csv(out/f"{name}.csv",rows)
    if bench:
        write_csv(out/"fast_bound_benchmark.csv",bench)
    plots(out,accuracy,bench)
    report = {
        "mathematical_results":proof,"certification_status":"empirically_validated_only",
        "field_relative_accuracy_tolerance":FIELD_TOL,"primary_accuracy_gate_pass":primary_pass,
        "cases":results,"benchmark":bench,"identical_amplitude_and_phase_conventions":True,
        "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                       "platform":platform.platform(),"processor":platform.processor(),
                       "thread_settings":{key:os.environ.get(key) for key in
                                          ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}},
        "limits":[
            "The sampled Chebyshev uniform error and observed rule differences are empirical estimates. "
            "No certified epsilon_L or validated floating-point/ROI norm enclosure was produced.",
            "Analytic moment envelopes concern continuous moments; their numerically evaluated ROI norms are not certified.",
            "Benchmark memory is a working-array budget, not measured peak process RSS.",
            "Identical nodes, normalized comparison measure, no phase alignment, and no amplitude fit.",
            "Each benchmark evaluates both Taylor orders. The 2D method includes direct positive remainder "
            "quadrature; boundary methods use the moment majorant.",
            "Cold benchmark includes setup, field and remainder evaluation, after selecting a rule. "
            "It excludes the separate accuracy audit and does not time a certified error budget.",
        ],
    }
    (out/"fast_bound_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Saved to {out.resolve()}")


if __name__=="__main__":
    main()
