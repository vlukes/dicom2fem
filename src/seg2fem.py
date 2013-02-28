#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentation data to FE mesh.

Example:

$ seg2fem.py -f brain_seg.mat
"""

import scipy.sparse as sps
import numpy as nm
from numpy.core import intc
from numpy.linalg import lapack_lite

# compatibility
try:
    import scipy as sc
    factorial = sc.factorial

except AttributeError:
    import scipy.misc as scm

    factorial = scm.factorial

def output(msg):
    print msg

def elems_q2t(el):

    nel, nnd = el.shape
    if nnd > 4:
        q2t = nm.array([[0, 2, 3, 6],
                        [0, 3, 7, 6],
                        [0, 7, 4, 6],
                        [0, 5, 6, 4],
                        [1, 5, 6, 0],
                        [1, 6, 2, 0]])

    else:
        q2t = nm.array([[0, 1, 2],
                        [0, 2, 3]])

    ns, nn = q2t.shape
    nel *= ns

    out = nm.zeros((nel, nn), dtype=nm.int32);

    for ii in range(ns):
        idxs = nm.arange(ii, nel, ns)

        out[idxs,:] = el[:, q2t[ii,:]]

    return nm.ascontiguousarray(out)

def smooth_mesh(mesh, n_iter=4, lam=0.6307, mu=-0.6347,
                weights=None, bconstr=True,
                volume_corr=False):
    """
    FE mesh smoothing.

    Based on:

    [1] Steven K. Boyd, Ralph Muller, Smooth surface meshing for automated
    finite element model generation from 3D image data, Journal of
    Biomechanics, Volume 39, Issue 7, 2006, Pages 1287-1295,
    ISSN 0021-9290, 10.1016/j.jbiomech.2005.03.006.
    (http://www.sciencedirect.com/science/article/pii/S0021929005001442)

    Parameters
    ----------
    mesh : mesh
        FE mesh.
    n_iter : integer, optional
        Number of iteration steps.
    lam : float, optional
        Smoothing factor, see [1].
    mu : float, optional
        Unshrinking factor, see [1].
    weights : array, optional
        Edge weights, see [1].
    bconstr: logical, optional
        Boundary constraints, if True only surface smoothing performed.
    volume_corr: logical, optional
        Correct volume after smoothing process.

    Returns
    -------
    coors : array
        Coordinates of mesh nodes.
    """

    def laplacian(coors, weights):

        n_nod = coors.shape[0]
        displ = (weights - sps.identity(n_nod)) * coors

        return displ

    def taubin(coors0, weights, lam, mu, n_iter):

        coors = coors0.copy()

        for ii in range(n_iter):
            displ = laplacian(coors, weights)
            if nm.mod(ii, 2) == 0:
                coors += lam * displ
            else:
                coors += mu * displ

        return coors

    def dets_fast(a):
        m = a.shape[0]
        n = a.shape[1]
        lapack_routine = lapack_lite.dgetrf
        pivots = nm.zeros((m, n), intc)
        flags = nm.arange(1, n + 1).reshape(1, -1)
        for i in xrange(m):
            tmp = a[i]
            lapack_routine(n, n, tmp, n, pivots[i], 0)
        sign = 1. - 2. * (nm.add.reduce(pivots != flags, axis=1) % 2)
        idx = nm.arange(n)
        d = a[:, idx, idx]
        absd = nm.absolute(d)
        sign *= nm.multiply.reduce(d / absd, axis=1)
        nm.log(absd, absd)
        logdet = nm.add.reduce(absd, axis=-1)

        return sign * nm.exp(logdet)

    def get_volume(el, nd):

        dim = nd.shape[1]
        nnd = el.shape[1]

        etype = '%d_%d' % (dim, nnd)
        if etype == '2_4' or etype == '3_8':
            el = elems_q2t(el)

        nel = el.shape[0]

        #bc = nm.zeros((dim, ), dtype=nm.double)
        mul = 1.0 / factorial(dim)
        if dim == 3:
            mul *= -1.0

        mtx = nm.ones((nel, dim + 1, dim + 1), dtype=nm.double)
        mtx[:,:,:-1] = nd[el,:]
        vols = mul * dets_fast(mtx.copy()) # copy() ???
        vol = vols.sum()
        bc = nm.dot(vols, mtx.sum(1)[:,:-1] / nnd)

        bc /= vol

        return vol, bc

    import time

    output('smoothing...')
    tt = time.clock()

    domain = Domain('mesh', mesh)

    n_nod = mesh.n_nod
    edges = domain.ed

    if weights is None:
        # initiate all vertices as inner - hierarchy = 2
        node_group = nm.ones((n_nod,), dtype=nm.int16) * 2
        # boundary vertices - set hierarchy = 4
        if bconstr:
            # get "nodes of surface"
            if domain.fa: # 3D.
                fa = domain.fa
            else:
                fa = domain.ed

            flag = fa.mark_surface_facets()
            ii = nm.where( flag > 0 )[0]
            aux = nm.unique(fa.facets[ii])
            if aux[0] == -1: # Triangular faces have -1 as 4. point.
                aux = aux[1:]

            node_group[aux] = 4

        # generate costs matrix
        mtx_ed = edges.mtx.tocoo()
        _, idxs = nm.unique(mtx_ed.row, return_index=True)
        aux = edges.facets[mtx_ed.col[idxs]]
        fc1 = aux[:,0]
        fc2 = aux[:,1]
        idxs = nm.where(node_group[fc2] >= node_group[fc1])
        rows1 = fc1[idxs]
        cols1 = fc2[idxs]
        idxs = nm.where(node_group[fc1] >= node_group[fc2])
        rows2 = fc2[idxs]
        cols2 = fc1[idxs]
        crows = nm.concatenate((rows1, rows2))
        ccols = nm.concatenate((cols1, cols2))
        costs = sps.coo_matrix((nm.ones_like(crows), (crows, ccols)),
                               shape=(n_nod, n_nod),
                               dtype=nm.double)

        # generate weights matrix
        idxs = range(n_nod)
        aux = sps.coo_matrix((1.0 / nm.asarray(costs.sum(1)).squeeze(),
                              (idxs, idxs)),
                             shape=(n_nod, n_nod),
                             dtype=nm.double)

        #aux.setdiag(1.0 / costs.sum(1))
        weights = (aux.tocsc() * costs.tocsc()).tocsr()

    coors = taubin(mesh.coors, weights, lam, mu, n_iter)

    output('...done in %.2f s' % (time.clock() - tt))

    if volume_corr:
        output('rescaling...')
        volume0, bc = get_volume(mesh.conns[0], mesh.coors)
        volume, _ = get_volume(mesh.conns[0], coors)

        scale = volume0 / volume
        output('scale factor: %.2f' % scale)

        coors = (coors - bc) * scale + bc

        output('...done in %.2f s' % (time.clock() - tt))

    return coors

def gen_mesh_from_voxels(voxels, dims, etype='q'):
    """
    Generate FE mesh from voxels (volumetric data).

    Parameters
    ----------
    voxels : array
        Voxel matrix, 1=material.
    dims : array
        Size of one voxel.
    etype : integer, optional
        'q' - quadrilateral or hexahedral elements
        't' - triangular or tetrahedral elements
    Returns
    -------
    mesh : Mesh instance
        Finite element mesh.
    """

    dims = dims.squeeze()
    dim = len(dims)
    nddims = nm.array(voxels.shape) + 2

    nodemtx = nm.zeros(nddims, dtype=nm.int32)

    if dim == 2:
        #iy, ix = nm.where(voxels.transpose())
        iy, ix = nm.where(voxels)
        nel = ix.shape[0]

        if etype == 'q':
            nodemtx[ix,iy] += 1
            nodemtx[ix + 1,iy] += 1
            nodemtx[ix + 1,iy + 1] += 1
            nodemtx[ix,iy + 1] += 1

        elif etype == 't':
            nodemtx[ix,iy] += 2
            nodemtx[ix + 1,iy] += 1
            nodemtx[ix + 1,iy + 1] += 2
            nodemtx[ix,iy + 1] += 1
            nel *= 2

    elif dim == 3:
        #iy, ix, iz = nm.where(voxels.transpose(1, 0, 2))
        iy, ix, iz = nm.where(voxels)
        nel = ix.shape[0]

        if etype == 'q':
            nodemtx[ix,iy,iz] += 1
            nodemtx[ix + 1,iy,iz] += 1
            nodemtx[ix + 1,iy + 1,iz] += 1
            nodemtx[ix,iy + 1,iz] += 1
            nodemtx[ix,iy,iz + 1] += 1
            nodemtx[ix + 1,iy,iz + 1] += 1
            nodemtx[ix + 1,iy + 1,iz + 1] += 1
            nodemtx[ix,iy + 1,iz + 1] += 1

        elif etype == 't':
            nodemtx[ix,iy,iz] += 6
            nodemtx[ix + 1,iy,iz] += 2
            nodemtx[ix + 1,iy + 1,iz] += 2
            nodemtx[ix,iy + 1,iz] += 2
            nodemtx[ix,iy,iz + 1] += 2
            nodemtx[ix + 1,iy,iz + 1] += 2
            nodemtx[ix + 1,iy + 1,iz + 1] += 6
            nodemtx[ix,iy + 1,iz + 1] += 2
            nel *= 6

    else:
        msg = 'incorrect voxel dimension! (%d)' % dim
        raise ValueError(msg)

    ndidx = nm.where(nodemtx)
    coors = nm.array(ndidx).transpose() * dims
    nnod = coors.shape[0]

    nodeid = -nm.ones(nddims, dtype=nm.int32)
    nodeid[ndidx] = nm.arange(nnod)

    # generate elements
    if dim == 2:
        elems = nm.array([nodeid[ix,iy],
                          nodeid[ix + 1,iy],
                          nodeid[ix + 1,iy + 1],
                          nodeid[ix,iy + 1]]).transpose()

    elif dim == 3:
        elems = nm.array([nodeid[ix,iy,iz],
                          nodeid[ix + 1,iy,iz],
                          nodeid[ix + 1,iy + 1,iz],
                          nodeid[ix,iy + 1,iz],
                          nodeid[ix,iy,iz + 1],
                          nodeid[ix + 1,iy,iz + 1],
                          nodeid[ix + 1,iy + 1,iz + 1],
                          nodeid[ix,iy + 1,iz + 1]]).transpose()

    if etype == 't':
        elems = elems_q2t(elems)

    eid = etype + str(dim)
    eltab = {'q2': 4, 'q3': 8, 't2': 3, 't3': 4}

    mesh = Mesh.from_data('voxel_data',
                          coors, nm.ones((nnod,), dtype=nm.int32),
                          {0: nm.ascontiguousarray(elems)},
                          {0: nm.ones((nel,), dtype=nm.int32)},
                          {0: '%d_%d' % (dim, eltab[eid])})

    return mesh

smooth_methods = {
    'none': None,
    'taubin': smooth_mesh,
    }

usage = '%prog [options]\n' + __doc__.rstrip()
help = {
    'in_file': 'input *.mat file with "data" field',
    'mode': '"seed" or "crop" mode',
    #'out_file': 'store the output matrix to the file',
    #'debug': 'run in debug mode',
    'test': 'run unit test',
}

def main():
    parser = OptionParser(description='Segmentation editor')
    parser.add_option('-f','--filename', action='store',
                      dest='in_filename', default=None,
                      help=help['in_file'])
    # parser.add_option('-d', '--debug', action='store_true',
    #                   dest='debug', help=help['debug'])
    parser.add_option('-m', '--mode', action='store',
                      dest='mode', default='seed', help=help['mode'])
    parser.add_option('-t', '--tests', action='store_true',
                      dest='unit_test', help=help['test'])
    # parser.add_option('-o', '--outputfile', action='store',
    #                   dest='out_filename', default='output.mat',
    #                   help=help['out_file'])
    (options, args) = parser.parse_args()

    # if options.tests:
    #     # hack for use argparse and unittest in one module
    #     sys.argv[1:]=[]
    #     unittest.main()
    #gen_mesh_from_voxels(voxels, dims, etype='q'):

    if options.in_filename is None:
        raise IOError('No input data!')

    else:
        dataraw = loadmat(options.in_filename,
                          variable_names=['data', 'voxelsizemm'])
        
    app = QApplication(sys.argv)
    pyed = QTSeedEditor(dataraw['data'],
                        mode=options.mode,
                        voxelVolume=np.prod(dataraw['voxelsizemm']))
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
