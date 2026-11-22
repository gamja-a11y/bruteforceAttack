"""Diagnostic 3D Meep cross-check for the area-preserving CIRCULAR (p=2) aperture.

This fixes a critical geometry issue: p=2 must use a CYLINDRICAL through-hole,
not an ellipsoidal air inclusion whose radius shrinks near the PEC faces.
This is deliberately NOT a new Fresnel error predictor. It measures whether the
incident field matches the uniform-plane-wave scalar assumption and separately
reports the residual FDTD-vs-scalar differences.

Requires `fresnel_shape_sensitivity.py` (the uploaded scalar experiment code)
in the same directory. Optional --mode audit works without Meep. Full FDTD
--mode run is computationally expensive for z=100 and should only be run after
reviewing the printed cell-size estimate and existing run diagnostics.

Examples:
  python meep_shape_diagnostic_v2.py --mode audit --csv shape_sensitivity_meep_A/shape_sensitivity.csv
  python meep_shape_diagnostic_v2.py --mode plan
  python meep_shape_diagnostic_v2.py --mode run --resolution 6 --out meep_caseA_v2
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from fresnel_shape_sensitivity import Experiment, aperture_rule, observation_rule, fields


def phase_align(field, reference):
    overlap = np.vdot(reference, field)
    return field * np.exp(-1j * np.angle(overlap)) if abs(overlap) else field


def relative_l2(field, reference):
    return float(np.linalg.norm(field-reference) / np.linalg.norm(reference))


def weighted_l2(field, reference, weights):
    return float(np.sqrt(np.sum(weights * abs(field-reference)**2) /
                         np.sum(weights * abs(reference)**2)))


def scalar_fields(exp, quad, x, y):
    arr, _, _, _ = fields(2., exp, quad, x, y)
    return arr[:, 0], arr[:, 1]


def inspect_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty CSV')
    print(f'Rows: {len(rows)}')
    for row in rows:
        print('p:', row.get('p'), 'resolution:', row.get('resolution'))
        for key in ('incident_nonuniformity', 'E', 'E_hat', 'moment_vs_meep', 'dE_dp', 'dE_hat_dp'):
            if key in row:
                print(f'  {key}: {float(row[key]):.9g}')
        if 'E' in row and 'E_hat' in row:
            print('  CAUTION: E compares Meep with Fresnel, while E_hat predicts RS-I vs Fresnel.')
            print('  They are NOT estimates of the same physical/model error.')
    print('No Meep rerun required for this audit. Raw complex fields were not saved in this CSV.')


def estimate(exp, resolution, pml, margin, z_src):
    # identical scale along all axes, and only a single aperture model
    half_xy = exp.radius + margin
    sx = sy = 2 * (half_xy + pml)
    zmin = z_src - margin - pml
    zmax = exp.distance + margin + pml
    sz = zmax - zmin
    voxels = (sx * resolution) * (sy * resolution) * (sz * resolution)
    return dict(cell=[sx,sy,sz], estimated_voxels=round(voxels),
                z_center=(zmin+zmax)/2, half_xy=half_xy)


def sampled_dft(sim, monitor, component, x, y):
    arr = np.squeeze(sim.get_dft_array(monitor, component, 0))
    xs,ys,_,_ = sim.get_array_metadata(dft_cell=monitor)
    xs,ys=np.asarray(xs).ravel(),np.asarray(ys).ravel()
    if arr.ndim != 2 or arr.shape != (xs.size,ys.size):
        raise RuntimeError(f'DFT axes unexpected: {arr.shape}, expected {(xs.size,ys.size)}. '
                           'Do not silently transpose square arrays.')
    if np.any(np.diff(xs) <= 0) or np.any(np.diff(ys) <= 0):
        raise RuntimeError('DFT coordinate axis not monotonically increasing')
    interpolate=RegularGridInterpolator((xs,ys),arr,bounds_error=False,fill_value=np.nan+0j)
    values=interpolate(np.column_stack((x,y)))
    if not np.all(np.isfinite(values)):
        raise RuntimeError('Observation or aperture samples outside DFT plane')
    return values


def meep_run(exp,args,xobs,yobs,xap,yap):
    import meep as mp
    fcen=1/exp.wavelength
    pml=args.pml
    margin=args.margin
    z_src=-max(args.screen_thickness/2+0.7,0.8)
    info=estimate(exp,args.resolution,pml,margin,z_src)
    sx,sy,sz=info['cell']; zc=info['z_center']; half_xy=info['half_xy']
    print('Cell:',info,'(per FDTD run; two runs required)',flush=True)
    # The source is a finite sheet. Its edge diffraction is an unavoidable
    # diagnostic: don't assume this is a perfect plane wave.
    def simulate(with_screen):
        geometry=[]
        if with_screen:
            geometry=[mp.Block(size=mp.Vector3(sx,sy,args.screen_thickness),
                               center=mp.Vector3(),material=mp.metal),
                      mp.Cylinder(radius=exp.radius,
                                  height=args.screen_thickness+2/args.resolution,
                                  axis=mp.Vector3(0,0,1),
                                  center=mp.Vector3(),material=mp.air)]
        source=mp.Source(mp.GaussianSource(frequency=fcen,fwidth=args.bandwidth*fcen),
                         component=mp.Ex,center=mp.Vector3(0,0,z_src),
                         size=mp.Vector3(2*half_xy,2*half_xy,0))
        sim=mp.Simulation(cell_size=mp.Vector3(sx,sy,sz),
                          geometry_center=mp.Vector3(0,0,zc),
                          geometry=geometry,
                          boundary_layers=[mp.PML(pml)],
                          sources=[source],resolution=args.resolution)
        # Measure SAME frequencies and lateral points on both runs.
        mon_out=sim.add_dft_fields([mp.Ex],fcen,0,1,
                  center=mp.Vector3(0,0,exp.distance),
                  size=mp.Vector3(2*half_xy,2*half_xy,0))
        mon_ap=None
        if not with_screen:
            mon_ap=sim.add_dft_fields([mp.Ex],fcen,0,1,
                   center=mp.Vector3(),size=mp.Vector3(2*half_xy,2*half_xy,0))
        sim.run(until=args.runtime)
        uobs=sampled_dft(sim,mon_out,mp.Ex,xobs,yobs)
        uap=sampled_dft(sim,mon_ap,mp.Ex,xap,yap) if mon_ap is not None else None
        sim.reset_meep()
        return uobs,uap
    print('Reference (no-screen) run',flush=True)
    inc_obs, inc_ap=simulate(False)
    print('Screen with circular hole run',flush=True)
    transmitted,_=simulate(True)
    return inc_obs,inc_ap,transmitted,info


def run(args):
    exp=Experiment()
    if not np.isclose(exp.aspect_ratio,1.):
        raise ValueError('This diagnostic targets only the p=2 circular baseline')
    z_src=-max(args.screen_thickness/2+0.7,0.8)
    info=estimate(exp,args.resolution,args.pml,args.margin,z_src)
    if args.mode=='plan':
        print(json.dumps(info,indent=2))
        print('Two 3D FDTD runs will be required. This does NOT cure PEC/vector model differences.')
        return
    # observation_rule uses a symmetric quarter-disk; quadrature weights include
    # radial Jacobian in the weights, not a naive uniform Cartesian grid.
    xobs,yobs,wobs=observation_rule(exp,args.observation_order)
    xap,yap,wap=aperture_rule(2.,exp,args.aperture_order)
    if args.mode=='run' and args.runtime < exp.distance - z_src + 20:
        raise ValueError(
            f'--runtime {args.runtime} is too short; wave flight time is '
            f'{exp.distance-z_src:.1f}, choose >= {exp.distance-z_src+20:.1f}'
        )
    if args.mode=='run' and args.resolution < 6:
        raise ValueError('Resolution <6 is not meaningful for this diagnostic')
    inc_obs,inc_ap,transmitted,info=meep_run(exp,args,xobs,yobs,xap,yap)
    # Average incident field on *aperture* samples using quadrature weights.
    e0=np.sum(wap*inc_ap)/np.sum(wap)
    if abs(e0) < 1e-14:
        raise RuntimeError('Incident field on aperture is effectively zero')
    u_meep=transmitted/e0
    u_inc_ap=inc_ap/e0
    u_inc_obs=inc_obs/e0
    # On each observation point the scalar theory predicts a plane-wave-driven
    # field; compare raw AND phase-aligned. Don't automatically scale amplitudes
    # to force agreement; this would obscure transmission errors.
    u_rs,u_fr=scalar_fields(exp,args.aperture_order,xobs,yobs)
    incident_rms=float(np.sqrt(np.sum(wap*abs(u_inc_ap-1)**2)/np.sum(wap)))
    incident_max=float(np.max(abs(u_inc_ap-1)))
    valid={
        'incident_aperture_rms_complex_nonuniformity':incident_rms,
        'incident_aperture_max_complex_nonuniformity':incident_max,
        'scalar_RS_vs_Fresnel':weighted_l2(u_fr,u_rs,wobs),
        'Meep_vs_RS_raw':weighted_l2(u_meep,u_rs,wobs),
        'Meep_vs_RS_phase_aligned':weighted_l2(phase_align(u_meep,u_rs),u_rs,wobs),
        'Meep_vs_Fresnel_phase_aligned':weighted_l2(phase_align(u_meep,u_fr),u_fr,wobs),
        'Meep_vs_Fresnel_intensity_L1':float(np.sum(wobs*abs(abs(u_meep)**2-abs(u_fr)**2)) /
                                                np.sum(wobs*abs(u_fr)**2)),
        'resolution':args.resolution,
        'p':2.,
        'Nf_eff':exp.fresnel_number,
        'theta_ap':exp.theta_ap,
        'physical_screen_thickness':args.screen_thickness,
        'quadrature_order':args.aperture_order,
        'observation_order':args.observation_order,
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'complex_fields.npz',
        xobs=xobs,yobs=yobs,wobs=wobs,
        xap=xap,yap=yap,wap=wap,
        incident_aperture=inc_ap,incident_observation=inc_obs,
        meep=u_meep,rs=u_rs,fresnel=u_fr)
    (out/'diagnostic_metrics.json').write_text(json.dumps(valid,indent=2),encoding='utf8')
    for k,v in valid.items():print(k,v)
    if incident_rms > .02:
        print('WARNING: incident field across aperture is not spatially uniform at 2% RMS level.')
        print('Do NOT interpret FDTD-vs-scalar residual as purely Fresnel approximation error.')
    print('Saved:',out.resolve())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['audit','plan','run'],default='audit')
    parser.add_argument('--csv',type=Path,default=Path('shape_sensitivity_meep_A/shape_sensitivity.csv'))
    parser.add_argument('--out',type=Path,default=Path('meep_caseA_diagnostic_v2'))
    parser.add_argument('--resolution',type=int,default=6)
    parser.add_argument('--runtime',type=float,default=150.)
    parser.add_argument('--pml',type=float,default=1.)
    parser.add_argument('--margin',type=float,default=0.8)
    parser.add_argument('--screen-thickness',type=float,default=.2)
    parser.add_argument('--bandwidth',type=float,default=.25)
    parser.add_argument('--aperture-order',type=int,default=20)
    parser.add_argument('--observation-order',type=int,default=12)
    args=parser.parse_args()
    if args.mode=='audit':inspect_csv(args.csv)
    else:run(args)

if __name__=='__main__':main()
