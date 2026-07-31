"""VENDORED verbatim from the design pass: NAMES/BONDS tables and the NeRF builder.

Only ``NAMES``, ``BONDS`` and ``build_positions`` are used, via :mod:`alanine.system`.
Do NOT use ``build_positions`` for umbrella seeding -- see the alanine.system docstring.
"""
import numpy as np, openmm as mm, openmm.app as app, openmm.unit as u

def nerf(a, b, c, r, theta, phi):
    theta = np.radians(theta); phi = np.radians(phi)
    bc = c - b; bc = bc/np.linalg.norm(bc)
    n = np.cross(b - a, bc); n = n/np.linalg.norm(n)
    m = np.cross(n, bc)
    return c + (-r*np.cos(theta))*bc + (r*np.sin(theta)*np.cos(phi))*m + (r*np.sin(theta)*np.sin(phi))*n

# amber14 order:  ACE(HH31 CH3 HH32 HH33 C O) ALA(N H CA HA CB HB1 HB2 HB3 C O) NME(N H CH3 HH31 HH32 HH33)
NAMES = [('ACE','HH31','H'),('ACE','CH3','C'),('ACE','HH32','H'),('ACE','HH33','H'),('ACE','C','C'),('ACE','O','O'),
         ('ALA','N','N'),('ALA','H','H'),('ALA','CA','C'),('ALA','HA','H'),('ALA','CB','C'),
         ('ALA','HB1','H'),('ALA','HB2','H'),('ALA','HB3','H'),('ALA','C','C'),('ALA','O','O'),
         ('NME','N','N'),('NME','H','H'),('NME','CH3','C'),('NME','HH31','H'),('NME','HH32','H'),('NME','HH33','H')]
BONDS = [(0,1),(1,2),(1,3),(1,4),(4,5),(4,6),(6,7),(6,8),(8,9),(8,10),(10,11),(10,12),(10,13),
         (8,14),(14,15),(14,16),(16,17),(16,18),(18,19),(18,20),(18,21)]
PHI_ATOMS = (4, 6, 8, 14)     # C(ACE) N CA C(ALA)
PSI_ATOMS = (6, 8, 14, 16)    # N CA C(ALA) N(NME)

def build_positions(phi_deg=-80.0, psi_deg=80.0, cb_offset=-120.0):
    X = np.zeros((22,3))
    X[1] = [0.,0.,0.]                                   # ACE CH3
    X[4] = [1.522,0.,0.]                                # ACE C
    a = np.radians(116.6)
    X[6] = X[4] + 1.335*np.array([-np.cos(a), np.sin(a), 0.])   # N
    X[8]  = nerf(X[1], X[4], X[6], 1.449, 121.9, 180.0)          # CA (omega trans)
    X[14] = nerf(X[4], X[6], X[8], 1.522, 110.4, phi_deg)        # ALA C
    X[16] = nerf(X[6], X[8], X[14], 1.335, 116.6, psi_deg)       # NME N
    X[18] = nerf(X[8], X[14], X[16], 1.449, 121.9, 180.0)        # NME CH3
    X[5]  = nerf(X[8], X[6], X[4], 1.229, 122.9, 180.0)          # ACE O  (anti to CA)
    X[15] = nerf(X[18], X[16], X[14], 1.229, 122.9, 180.0)       # ALA O
    X[10] = nerf(X[4], X[6], X[8], 1.526, 110.5, phi_deg + cb_offset)   # CB
    X[9]  = nerf(X[4], X[6], X[8], 1.090, 108.0, phi_deg - cb_offset)   # HA
    X[7]  = nerf(X[1], X[4], X[6], 1.010, 119.0, 0.0)            # amide H (ALA N)
    X[17] = nerf(X[8], X[14], X[16], 1.010, 119.0, 0.0)          # amide H (NME N)
    for k,(i0,i1,i2) in enumerate([(0,2,3)]):                    # ACE methyl H
        for j,idx in enumerate((0,2,3)):
            X[idx] = nerf(X[6], X[4], X[1], 1.090, 109.5, 60.0 + 120.0*j)
    for j,idx in enumerate((11,12,13)):                          # CB methyl H
        X[idx] = nerf(X[6], X[8], X[10], 1.090, 109.5, 60.0 + 120.0*j)
    for j,idx in enumerate((19,20,21)):                          # NME methyl H
        X[idx] = nerf(X[14], X[16], X[18], 1.090, 109.5, 60.0 + 120.0*j)
    return X

def topology():
    top = app.Topology(); ch = top.addChain(); E = app.element
    res = {}
    atoms = []
    for i,(rn, an, el) in enumerate(NAMES):
        if rn not in res: res[rn] = top.addResidue(rn, ch)
        atoms.append(top.addAtom(an, getattr(E, {'H':'hydrogen','C':'carbon','N':'nitrogen','O':'oxygen'}[el]), res[rn]))
    for i,j in BONDS: top.addBond(atoms[i], atoms[j])
    return top

def dihedral(x, idx):
    p0,p1,p2,p3 = x[list(idx)]
    b0 = p0-p1; b1 = p2-p1; b2 = p3-p2
    b1n = b1/np.linalg.norm(b1)
    v = b0 - np.dot(b0,b1n)*b1n
    w = b2 - np.dot(b2,b1n)*b1n
    return np.degrees(np.arctan2(np.dot(np.cross(b1n,v),w), np.dot(v,w)))

def chirality(x):
    """CORN rule: >0 means L-amino acid ( ((C-CA) x (CB-CA)) . (HA-CA) > 0 )."""
    CA,C,CB,HA = x[8], x[14], x[10], x[9]
    return float(np.dot(np.cross(C-CA, CB-CA), HA-CA))

def make_system(ff_files=('amber14/protein.ff14SB.xml',), constraints=None, hmr=None):
    top = topology()
    ff = app.ForceField(*ff_files)
    kw = dict(nonbondedMethod=app.NoCutoff, constraints=constraints, rigidWater=False,
              removeCMMotion=False)
    if hmr is not None: kw['hydrogenMass'] = hmr
    return ff, top, ff.createSystem(top, **kw)
