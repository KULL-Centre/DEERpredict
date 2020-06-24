#!/usr/bin/env python

import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import sys


ref_pdb='0.pdb'
ref_u = mda.Universe(ref_pdb)


def calcDih(res):
    """

    Args:
        res:

    Returns:

    """
    x1Sel = res['N'] + res['CA'] + res['CB'] + res['CG1']
    x2Sel = res['CA'] + res['CB'] + res['CG1'] + res['CD1']

    coords = x1Sel.coordinates()
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[1]
    v3 = coords[3] - coords[2]
    v4 = np.cross(v1, v2)
    v5 = np.cross(v2, v3)
    LHS = np.dot(np.cross(v4, v5), np.divide(v2, np.linalg.norm(v2)))
    RHS = np.dot(v4, v5)
    dih = np.arctan2(LHS, RHS)
    x1 = np.rad2deg(dih)
    # x1= np.arctan2(np.dot(v3, v4), np.dot(v3, v5) * np.sqrt(np.dot(v2, v2)))

    coords = x2Sel.coordinates()
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[1]
    v3 = coords[3] - coords[2]
    v4 = np.cross(v1, v2)
    v5 = np.cross(v2, v3)
    LHS = np.dot(np.cross(v4, v5), np.divide(v2, np.linalg.norm(v2)))
    RHS = np.dot(v4, v5)
    dih = np.arctan2(LHS, RHS)
    x2 = np.rad2deg(dih)
    # x2= np.arctan2(np.dot(v3, v4), np.dot(v3, v5) * np.sqrt(np.dot(v2, v2)))

    return x1, x2


def getCoor(res):
    x1Sel = res['CG1']
    x2Sel = res['CD1']
    coords1 = x1Sel.coordinates()
    coords2 = x2Sel.coordinates()
    vec = coords2-coords1
    print(vec)

    return vec[0], vec[1], vec[2]


def loopPDB(sysarg, ref_u, dihDict, xyz):
    for p,pdb in enumerate(sysarg[1:]):
        new_u = mda.Universe(pdb)
        sel = 'name CA'
        align.alignto(new_u, ref_u, select=sel)
        for res in u.SYSTEM.ILE:
            dihDict[res.num].append(calcDih(res))
            xx,yy,zz = getCoor(res)
            xyz[res.resnum].append([xx,yy,zz])

    for key in xyz:
        S2[key]=calcS2(xyz[key])

    return dihDict, S2

def calcS2(xyz):
    x = [x[0] for x in xyz]
    y = [x[1] for x in xyz]
    z = [x[2] for x in xyz]
    S2 = 1.5 * np.mean(x**2)**2 + np.mean(y**2)**2 + np.mean(z**2)**2 + 2 * np.mean(x*y)**2 + 2* np.mean(x*z)**2 +2 * np.mean(y*z)**2 - 0.5
    return S2


if __name__ == '__main__':
    xyz={}
    dihDict = {}
    for res in u.SYSTEM.ILE:
        dihDict[res.resnum] = []
        xyz[res.resnum] = []

    RMSD=loopPDB(sys.argv, ref_u)
    with open('rmsd_all.dat','w') as out:
        for key in RMSD:
            out.write('%s %f \n'%(key,RMSD[key]))