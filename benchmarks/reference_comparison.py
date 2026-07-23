"""Compare identical analytical workloads across Python and Rust CPD implementations."""
from __future__ import annotations
import csv, importlib, os, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CRATE=ROOT/"rustcpd"
RUST=ROOT/"target"/"release"/"benchmark"

def cloud(n):
    z=np.arange(1,n+1,dtype=np.float64)
    return np.column_stack((np.sin(z*.037)*1.7+.0001*z,np.cos(z*.023)*.9,np.sin(z*.011)*np.cos(z*.007)))

def run_python(method,n,iterations,repeats):
    package=os.environ.get("CPD_REFERENCE_PACKAGE")
    if not package: raise RuntimeError("Set CPD_REFERENCE_PACKAGE to the Python reference package name")
    api=importlib.import_module(package); utility=importlib.import_module(f"{package}.utility")
    AtlasRegistration=api.AtlasRegistration; DeformableRegistration=api.DeformableRegistration; RigidRegistration=api.RigidRegistration
    gaussian_kernel=utility.gaussian_kernel
    y=cloud(n);times=[];error=0;sparse=method.endswith("_sparse");method=method.removesuffix("_sparse")
    for _ in range(repeats):
        start=time.perf_counter()
        if method=="rigid":
            a=.12;r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
            x=1.04*(y@r)+[.12,-.08,.04]
            result=RigidRegistration(X=x,Y=y,use_kdtree=sparse,k=10,max_iterations=iterations,tolerance=0).register()[0]
        elif method=="deformable":
            g=gaussian_kernel(y,beta=1.5);w=np.fromfunction(lambda i,j:.003*np.sin((i*3+j+1)*.041),(n,3))
            x=y+g@w
            result=DeformableRegistration(X=x,Y=y,alpha=2,beta=1.5,low_rank=False,use_kdtree=sparse,k=10,dtype=np.float64,max_iterations=iterations,tolerance=0).register()[0]
        else:
            rank=12;modes=np.fromfunction(lambda i,k:.01*np.sin((i+1+k*7)*.019),(y.size,rank));coeff=.2*np.sin((np.arange(rank)+1)*.7)
            x=y+(modes@coeff).reshape(y.shape)
            result=AtlasRegistration(X=x,Y=y,U=modes,eigenvalues=1/(np.arange(rank)+1),lambda_reg=.1,optimize_similarity=False,use_kdtree=sparse,k=10,kdtree_radius_scale=1e-6,dtype=np.float64,max_iterations=iterations,tolerance=0).register()[0]
        times.append(time.perf_counter()-start);error=np.sqrt(np.mean((result-x)**2))
    return sorted(times)[len(times)//2],error

def main():
    subprocess.run(["cargo","build","--release","--bin","benchmark"],cwd=CRATE,check=True)
    cases=[("rigid",300,10), ("rigid",1000,10), ("rigid_sparse",1000,10), ("deformable",200,10), ("deformable_sparse",200,10), ("atlas",1000,10), ("atlas_sparse",1000,10)]
    print("method,n,iterations,python_seconds,rust_seconds,speedup,python_rms,rust_rms")
    for method,n,iterations in cases:
        py_time,py_error=run_python(method,n,iterations,7)
        output=subprocess.check_output([RUST,method,str(n),str(iterations),"7"],text=True).strip().splitlines()[-1]
        row=next(csv.reader([output]));rust_time=float(row[4]);rust_error=float(row[5])
        print(f"{method},{n},{iterations},{py_time:.9f},{rust_time:.9f},{py_time/rust_time:.3f},{py_error:.12e},{rust_error:.12e}")

if __name__=="__main__":main()
