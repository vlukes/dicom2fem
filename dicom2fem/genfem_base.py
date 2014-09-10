import numpy as nm

def set_nodemtx(mtx, idxs, etype):

    dim = len(idxs)
    if dim == 2:
        ix, iy = idxs

        if etype == 'q':
            mtx[ix,iy] += 1
            mtx[ix + 1,iy] += 1
            mtx[ix + 1,iy + 1] += 1
            mtx[ix,iy + 1] += 1

        elif etype == 't':
            mtx[ix,iy] += 2
            mtx[ix + 1,iy] += 1
            mtx[ix + 1,iy + 1] += 2
            mtx[ix,iy + 1] += 1

    elif dim == 3:
        ix, iy, iz = idxs

        if etype == 'q':
            mtx[ix,iy,iz] += 1
            mtx[ix + 1,iy,iz] += 1
            mtx[ix + 1,iy + 1,iz] += 1
            mtx[ix,iy + 1,iz] += 1
            mtx[ix,iy,iz + 1] += 1
            mtx[ix + 1,iy,iz + 1] += 1
            mtx[ix + 1,iy + 1,iz + 1] += 1
            mtx[ix,iy + 1,iz + 1] += 1

        elif etype == 't':
            mtx[ix,iy,iz] += 6
            mtx[ix + 1,iy,iz] += 2
            mtx[ix + 1,iy + 1,iz] += 2
            mtx[ix,iy + 1,iz] += 2
            mtx[ix,iy,iz + 1] += 2
            mtx[ix + 1,iy,iz + 1] += 2
            mtx[ix + 1,iy + 1,iz + 1] += 6
            mtx[ix,iy + 1,iz + 1] += 2

    else:
        msg = 'incorrect voxel dimension! (%d)' % dim
        raise ValueError(msg)

edge_tab = {
    '2_3': nm.array([[0,1],
                     [1,2],
                     [2,0]]),
    '2_4': nm.array([[0,1],
                     [1,2],
                     [2,3],
                     [3,0]]),
    '3_4': nm.array([[0,1],
                     [1,2],
                     [2,0],
                     [0,3],
                     [1,3],
                     [2,3]]),
    '3_8': nm.array([[0,1],
                     [1,2],
                     [2,3],
                     [3,0],
                     [4,5],
                     [5,6],
                     [6,7],
                     [7,4],
                     [0,4],
                     [1,5],
                     [2,6],
                     [3,7]]),
}

face_tab = {
    '3_4': nm.array([[0,1,2],
                     [0,1,3],
                     [1,2,3],
                     [0,3,2]]),
    '3_8': nm.array([[0,1,2,3],
                     [0,1,5,4],
                     [1,2,6,5],
                     [2,3,7,6],
                     [3,0,4,7],
                     [4,5,6,7]]),
}

def unique_rows(a):
    a.sort(axis=1)
    order = nm.lexsort(a.T)
    a = a[order]
    diff = nm.diff(a, axis=0)
    ui = nm.ones(len(a), dtype=nm.bool)
    ui[1:] = (diff != 0).any(axis=1)
    uio = nm.zeros_like(ui)
    uio[order] = ui

    sio = nm.ones_like(ui)
    ii = nm.where(nm.invert(ui))
    ui[(ii[0] - 1)] = False
    sio[order] = ui

    return uio, sio

def get_snodes_uedges(conns, etype):

    if etype[0] == '2':
        fci = edge_tab[etype]

    else:
        fci = face_tab[etype]

    nel = conns.shape[0]
    nnd = nm.max(conns) + 1
    nfc = fci.shape[0]
    nnpdfc = fci.shape[1]

    faces = conns[:,fci].reshape((nel * nfc, nnpdfc))
    ufci, sfci = unique_rows(faces)
    ufaces = faces[ufci]
    sfaces = faces[sfci]
    snodes = nm.zeros((nnd,), dtype=nm.bool)
    for ii in sfaces.T:
        snodes[(ii, )] = True

    sndi = nm.where(snodes)

    if etype[0] == '2':
        uedges = ufaces

    else:
        edi = edge_tab[etype]
        ned = edi.shape[0]
        edges = conns[:,edi].reshape((nel * ned, 2))
        uedi, _ = unique_rows(edges)
        uedges = edges[uedi]

    return sndi, uedges
