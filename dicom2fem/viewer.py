#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple VTK Viewer.

Example:

$ dicom2fem_view_mesh.py head.vtk
"""
from optparse import OptionParser
import sys
import pyvista as pv

usage = '%prog file.vtk\n' + __doc__.rstrip()


def view_mesh(fname):
    mesh = pv.read(fname)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, color=[222, 135, 135],
                     edge_color=[233, 175, 175], edge_opacity=0.2)
    plotter.add_axes()
    plotter.show(title=f'dicom2fem - mesh viewer: {fname}',
                 interactive=True)


def main():
    parser = OptionParser(usage=usage)
    _, args = parser.parse_args()

    if len(args) < 1:
        raise IOError('No VTK data!')

    view_mesh(args[0])


if __name__ == "__main__":
    main()
