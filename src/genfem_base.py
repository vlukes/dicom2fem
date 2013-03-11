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
