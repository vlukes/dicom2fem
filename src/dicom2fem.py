#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DICOM2FEM - organ segmentation and FE model generator

Example:

$ dicom2fem -d sample_data
"""

# import unittest
from optparse import OptionParser
from scipy.io import loadmat, savemat
from scipy import ndimage
import numpy as np
import sys
import os

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
     QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup

sys.path.append("./pyseg_base/src/")

import dcmreaddata as dcmreader
from seed_editor_qt import QTSeedEditor
import pycat
from meshio import supported_capabilities, supported_formats, MeshIO

from viewer import QVTKViewer

inv_supported_formats = dict(zip(supported_formats.values(), supported_formats.keys()))
smooth_methods = ['marching cubes', 'taubin']

elem_tab = {'2_3': 'triangles', '3_4': 'tetrahedrons',
            '2_4': 'quads', '3_8': 'hexahedrons'}

class MainWindow(QMainWindow):
    
    def __init__(self, dcmdir=None):
        QMainWindow.__init__(self)

        self.dcmdir = dcmdir
        self.dcm_3Ddata = None
        self.dcm_metadata = None
        self.dcm_zoom = None
        self.dcm_offsetmm = np.array([0,0,0])
        self.voxel_volume = 0.0
        self.voxel_sizemm = None
        self.segmentation_seeds = None
        self.segmentation_data = None
        self.mesh_data = None
        self.mesh_out_format = 'vtk'
        self.mesh_smooth_method = smooth_methods[0]
        self.initUI()
        
    def initUI(self):               

        cw = QWidget()
        self.setCentralWidget(cw)
        grid = QGridLayout()
        grid.setSpacing(15)
        
        # status bar
        self.statusBar().showMessage('Ready')

        font_label = QFont()
        font_label.setBold(True)

        ################ dicom reader
        rstart = 0
        text_dcm = QLabel('DICOM reader') 
        text_dcm.setFont(font_label)
        self.text_dcm_dir = QLabel('DICOM dir:')
        self.text_dcm_data = QLabel('DICOM data:')
        self.text_dcm_out = QLabel('output file:')
        grid.addWidget(text_dcm, rstart + 0, 1, 1, 4)
        grid.addWidget(self.text_dcm_dir, rstart + 1, 1, 1, 4)
        grid.addWidget(self.text_dcm_data, rstart + 2, 1, 1, 4)
        grid.addWidget(self.text_dcm_out, rstart + 3, 1, 1, 4)

        btn_dcmdir = QPushButton("Load DICOM", self)
        btn_dcmdir.clicked.connect(self.loadDcmDir)
        btn_dcmred = QPushButton("Reduce", self)
        btn_dcmred.clicked.connect(self.reduceDcm)
        btn_dcmcrop = QPushButton("Crop", self)
        btn_dcmcrop.clicked.connect(self.cropDcm)
        btn_dcmsave = QPushButton("Save DCM", self)
        btn_dcmsave.clicked.connect(self.saveDcm)
        grid.addWidget(btn_dcmdir, rstart + 4, 1)
        grid.addWidget(btn_dcmred, rstart + 4, 2)
        grid.addWidget(btn_dcmcrop, rstart + 4, 3)
        grid.addWidget(btn_dcmsave, rstart + 4, 4)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        grid.addWidget(hr, rstart + 5, 0, 1, 6)

        ################ segmentation
        rstart = 6
        text_seg = QLabel('Segmentation') 
        text_seg.setFont(font_label)
        self.text_seg_in = QLabel('input data:')
        self.text_seg_data = QLabel('segment. data:')
        self.text_seg_out = QLabel('output file:')
        grid.addWidget(text_seg, rstart + 0, 1)
        grid.addWidget(self.text_seg_in, rstart + 1, 1, 1, 4)
        grid.addWidget(self.text_seg_data, rstart + 2, 1, 1, 4)
        grid.addWidget(self.text_seg_out, rstart + 3, 1, 1, 4)

        btn_segload = QPushButton("Load DCM", self)
        btn_segload.clicked.connect(self.loadDcm)
        btn_segauto = QPushButton("Automatic seg.", self)
        btn_segauto.clicked.connect(self.autoSeg)
        btn_segman = QPushButton("Manual seg.", self)
        btn_segman.clicked.connect(self.manualSeg)
        btn_segsave = QPushButton("Save SEG", self)
        btn_segsave.clicked.connect(self.saveSeg)
        grid.addWidget(btn_segload, rstart + 4, 1)
        grid.addWidget(btn_segauto, rstart + 4, 2)
        grid.addWidget(btn_segman, rstart + 4, 3)
        grid.addWidget(btn_segsave, rstart + 4, 4)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        grid.addWidget(hr, rstart + 5, 0, 1, 6)

        ################ mesh gen.
        rstart = 12
        text_mesh = QLabel('Mesh generation') 
        text_mesh.setFont(font_label)
        self.text_mesh_in = QLabel('input data:')
        self.text_mesh_data = QLabel('mesh data:')
        self.text_mesh_out = QLabel('output file:')
        grid.addWidget(text_mesh, rstart + 0, 1)
        grid.addWidget(self.text_mesh_in, rstart + 1, 1, 1, 4)
        grid.addWidget(self.text_mesh_data, rstart + 2, 1, 1, 4)
        grid.addWidget(self.text_mesh_out, rstart + 3, 1, 1, 4)

        btn_meshload = QPushButton("Load SEG", self)
        btn_meshload.clicked.connect(self.loadSeg)
        btn_meshsave = QPushButton("Save MESH", self)
        btn_meshsave.clicked.connect(self.saveMesh)
        btn_meshgener = QPushButton("Generate", self)
        btn_meshgener.clicked.connect(self.generMesh)
        btn_meshsmooth = QPushButton("Smooth", self)
        btn_meshsmooth.clicked.connect(self.smoothMesh)
        btn_meshview = QPushButton("View", self)
        btn_meshview.clicked.connect(self.viewMesh)
        grid.addWidget(btn_meshload, rstart + 4, 1)
        grid.addWidget(btn_meshgener, rstart + 4, 2)
        grid.addWidget(btn_meshsmooth, rstart + 4, 3)
        grid.addWidget(btn_meshsave, rstart + 4, 4)
        grid.addWidget(btn_meshview, rstart + 8, 3, 1, 2)
        
        text_mesh_mesh = QLabel('mesh:')
        text_mesh_elements = QLabel('elements:')
        text_mesh_smooth = QLabel('smooth method:')
        text_mesh_output = QLabel('output format:')
        grid.addWidget(text_mesh_mesh, rstart + 6, 1)
        grid.addWidget(text_mesh_elements, rstart + 6, 2)
        grid.addWidget(text_mesh_smooth, rstart + 6, 3)
        grid.addWidget(text_mesh_output, rstart + 6, 4)
        
        rbtn_mesh_mesh_surf = QRadioButton('surface')
        rbtn_mesh_mesh_vol = QRadioButton('volume')
        grid.addWidget(rbtn_mesh_mesh_surf, rstart + 7, 1)
        grid.addWidget(rbtn_mesh_mesh_vol, rstart + 8, 1)
        self.rbtng_mesh_mesh = QButtonGroup(self)
        self.rbtng_mesh_mesh.addButton(rbtn_mesh_mesh_surf, 1)
        self.rbtng_mesh_mesh.addButton(rbtn_mesh_mesh_vol, 2)
        rbtn_mesh_mesh_vol.setChecked(True)

        rbtn_mesh_elements_3 = QRadioButton('tri/tetra')
        rbtn_mesh_elements_4 = QRadioButton('quad/hexa')
        grid.addWidget(rbtn_mesh_elements_3, rstart + 7, 2)
        grid.addWidget(rbtn_mesh_elements_4, rstart + 8, 2)
        self.rbtng_mesh_elements = QButtonGroup(self)
        self.rbtng_mesh_elements.addButton(rbtn_mesh_elements_3, 1)
        self.rbtng_mesh_elements.addButton(rbtn_mesh_elements_4, 2)
        rbtn_mesh_elements_4.setChecked(True)
        
        combo = QComboBox(self)
        combo.activated[str].connect(self.changeOut)
        
        supp_write = []
        for k, v in supported_capabilities.iteritems(): 
            if 'w' in v:
                supp_write.append(k)

        combo.addItems(supp_write)
        combo.setCurrentIndex(supp_write.index('vtk'))
        grid.addWidget(combo, rstart + 7, 4)

        combo2 = QComboBox(self)
        combo2.activated[str].connect(self.changeSmoothMethod)
        combo2.addItems(smooth_methods)
        combo2.setCurrentIndex(smooth_methods.index(self.mesh_smooth_method))
        grid.addWidget(combo2, rstart + 7, 3)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        grid.addWidget(hr, rstart + 9, 0, 1, 6)
        
        # quit
        btn_quit = QPushButton("Quit", self)
        btn_quit.clicked.connect(self.quit)
        grid.addWidget(btn_quit, 24, 2, 1, 2)
        
        cw.setLayout(grid)
        self.setWindowTitle('DICOM2FEM')    
        self.show()

    def quit(self, event):
        self.close()

    def setLabelText(self, obj, text):
        dlab = str(obj.text())
        obj.setText(dlab[:dlab.find(':')] + ': %s' % text)
        
    def getDcmInfo(self):        
        vsize = tuple([float(ii) for ii in self.voxel_sizemm])
        ret = ' %dx%dx%d,  %fx%fx%f mm' % (self.dcm_3Ddata.shape + vsize)
        
        return ret
              
    def setVoxelVolume(self, vxs):
        self.voxel_volume = np.prod(vxs)

    def loadDcmDir(self):
        self.statusBar().showMessage('Reading DICOM directory...')

        if self.dcmdir is None:
            self.dcmdir = dcmreader.get_dcmdir_qt(app=True)

        if self.dcmdir is not None:
            dcr = dcmreader.DicomReader(os.path.abspath(self.dcmdir))

        else:
            self.statusBar().showMessage('No DICOM directory specified!')
            return
        
        if dcr.validData():
            self.dcm_3Ddata = dcr.get_3Ddata()
            self.dcm_metadata = dcr.get_metaData()
            self.voxel_sizemm = np.array(self.dcm_metadata['voxelsizemm'])
            self.setVoxelVolume(self.voxel_sizemm)
            self.setLabelText(self.text_dcm_dir, self.dcmdir)
            self.setLabelText(self.text_dcm_data, self.getDcmInfo())
            self.statusBar().showMessage('Ready')
            self.setLabelText(self.text_seg_in, 'DICOM reader')
            
        else:
            self.statusBar().showMessage('No DICOM data in direcotry!')
            
    def reduceDcm(self, event=None, factor=None, default=0.25):
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        if factor is None:
            value, ok = QInputDialog.getText(self, 'Reduce DICOM data',
                                             'Reduce factor [0-1.0]:',
                                             text='%.2f' % default)
            if ok:
                vals = value.split(',')
                if len(vals) > 1:
                    factor = [float(ii) for ii in vals]

                else:
                    factor = float(value)

        if np.isscalar(factor):
            factor = [factor, factor, 1.0]

        self.dcm_zoom = np.array(factor[:3])
        for ii in factor:
           if ii < 0.0 or ii > 1.0:
               self.dcm_zoom = None
           
        if self.dcm_zoom is not None:
            self.dcm_3Ddata = ndimage.zoom(self.dcm_3Ddata, self.dcm_zoom,
                                           prefilter=False, mode='nearest')
            self.voxel_sizemm = self.voxel_sizemm / self.dcm_zoom
            self.setLabelText(self.text_dcm_data, self.getDcmInfo())

            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No valid reduce factor!')

    def cropDcm(self):
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        is_reduced = False
        shape = self.dcm_3Ddata.shape
        max12 = np.max(shape[:2])
        if max12 > 128:
            is_reduced = True
            czoom = [128.0 / max12] * 2
            if shape[2] > 32:
                czoom.append(32.0 / shape[2])

            else:
                czoom.append(1.0)

            czoom = np.array(czoom)
            reduced_3Ddata = ndimage.zoom(self.dcm_3Ddata, czoom,
                                          prefilter=False, mode='nearest')
        else:
            reduced_3Ddata = self.dcm_3Ddata
             
        pyed = QTSeedEditor(reduced_3Ddata, mode='crop')
        pyed.exec_()
        nzs = pyed.getSeeds().nonzero()

        if nzs is not None:
            cri = []
            for ii in range(3):
                if nzs[ii].shape[0] == 0:
                    nzs = None
                    break
                smin, smax = np.min(nzs[ii]), np.max(nzs[ii])
                if smin == smax:
                    nzs = None
                    break

                cri.append((smin, smax))            
            cri = np.array(cri)

            if is_reduced and nzs is not None:
                aux = (cri.T * (1.0 / czoom)).T
                cri[:,0] = np.floor(aux[:,0])
                cri[:,1] = np.ceil(aux[:,1])

        if nzs is not None:
            crop = self.dcm_3Ddata[cri[0][0]:(cri[0][1] + 1),
                                   cri[1][0]:(cri[1][1] + 1),
                                   cri[2][0]:(cri[2][1] + 1)] 
            self.dcm_3Ddata = np.ascontiguousarray(crop)
            self.dcm_offsetmm = cri[:,0] * self.voxel_sizemm

            self.setLabelText(self.text_dcm_data, self.getDcmInfo())
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No crop information!')

    def saveDcm(self, event=None, filename=None):
        if self.dcm_3Ddata is not None:
            if filename is None:
                filename = str(QFileDialog.getSaveFileName(self, 'Save DCM file',
                                                           filter='Files (*.dcm)'))
            if len(filename) > 0:
                self.statusBar().showMessage('Saving DICOM data...')

                savemat(filename, {'data': self.dcm_3Ddata,
                                   'voxelsizemm': self.voxel_sizemm,
                                   'offsetmm': self.dcm_offsetmm},
                                   appendmat=False)

                self.setLabelText(self.text_dcm_out, filename)
                self.statusBar().showMessage('Ready')
            
            else:
                self.statusBar().showMessage('No output file specified!')

        else:
            self.statusBar().showMessage('No DICOM data!')      

    def loadDcm(self, event=None, filename=None):
        if filename is None:
            filename = str(QFileDialog.getOpenFileName(self, 'Load DCM file',
                                                       filter='Files (*.dcm)'))

        if len(filename) > 0:
            self.statusBar().showMessage('Loading DICOM data...')

            data = loadmat(filename,
                           variable_names=['data', 'voxelsizemm', 'offsetmm'],
                           appendmat=False)
            
            self.dcm_3Ddata = data['data']
            self.voxel_sizemm = data['voxelsizemm']
            self.dcm_offsetmm = data['offsetmm'] 
            self.setVoxelVolume(self.voxel_sizemm.reshape((3,)))
            self.setLabelText(self.text_seg_in, filename)
            self.statusBar().showMessage('Ready')
            
        else:
            self.statusBar().showMessage('No input file specified!')
            
    def checkSegData(self):
        if self.segmentation_data is None:
            self.statusBar().showMessage('No SEG data!')
            return

        nzs = self.segmentation_data.nonzero()
        nn = nzs[0].shape[0]
        if nn > 0:
            aux = ' voxels = %d, volume = %.2e mm3' % (nn, nn * self.voxel_volume)
            self.setLabelText(self.text_seg_data, aux)
            self.setLabelText(self.text_mesh_in, 'segmentation data')
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('Zero SEG data!')

    def autoSeg(self):
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        igc = pycat.ImageGraphCut(self.dcm_3Ddata,
                                  voxelsize=self.voxel_sizemm)

        pyed = QTSeedEditor(self.dcm_3Ddata,
                            seeds=self.segmentation_seeds,
                            modeFun=igc.interactivity_loop,
                            voxelVolume=self.voxel_volume)
        pyed.exec_()

        self.segmentation_data = pyed.getContours()
        self.segmentation_seeds = pyed.getSeeds()
        self.checkSegData()

    def manualSeg(self): 
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        pyed = QTSeedEditor(self.dcm_3Ddata,
                            seeds=self.segmentation_data,
                            mode='draw',
                            voxelVolume=self.voxel_volume)
        pyed.exec_()

        self.segmentation_data = pyed.getSeeds()
        self.checkSegData()

    def saveSeg(self, event=None, filename=None):
        if self.segmentation_data is not None:
            if filename is None:
                filename = str(QFileDialog.getSaveFileName(self, 'Save SEG file',
                                                           filter='Files (*.seg)'))

            if len(filename) > 0:
                self.statusBar().showMessage('Saving segmentation data...')

                outdata = {'segdata': self.segmentation_data,
                            'voxelsizemm': self.voxel_sizemm,
                            'offsetmm': self.dcm_offsetmm}

                if self.segmentation_seeds is not None:
                    outdata['segseeds'] = self.segmentation_seeds

                savemat(filename, outdata, appendmat=False)
                self.setLabelText(self.text_seg_out, filename)
                self.statusBar().showMessage('Ready')

            else:
                self.statusBar().showMessage('No output file specified!')

        else:
            self.statusBar().showMessage('No segmentation data!')      

    def loadSeg(self, event=None, filename=None):
        if filename is None:
            filename = str(QFileDialog.getOpenFileName(self, 'Load SEG file',
                                                       filter='Files (*.seg)'))

        if len(filename) > 0:
            self.statusBar().showMessage('Loading segmentation data...')

            data = loadmat(filename,
                           variable_names=['segdata', 'segseeds',
                                           'voxelsizemm', 'offsetmm'],
                           appendmat=False)

            self.segmentation_data = data['segdata']
            if 'segseeds' in data:
                self.segmentation_seeds = data['segseeds']

            else:
                self.segmentation_seeds = None

            self.voxel_sizemm = data['voxelsizemm']
            self.dcm_offsetmm = data['offsetmm'] 
            self.setVoxelVolume(self.voxel_sizemm.reshape((3,)))
            self.setLabelText(self.text_mesh_in, filename)
            self.statusBar().showMessage('Ready')
            
        else:
            self.statusBar().showMessage('No input file specified!')

    def saveMesh(self, event=None, filename=None):
        if self.mesh_data is not None:
            if filename is None:
                file_ext = inv_supported_formats[self.mesh_out_format]
                filename = str(QFileDialog.getSaveFileName(self, 'Save MESH file',
                                                           filter='Files (*%s)' % file_ext))

            if len(filename) > 0:
                self.statusBar().showMessage('Saving mesh...')

                io = MeshIO.for_format(filename, format=self.mesh_out_format, writable=True)
                io.write(filename, self.mesh_data)

                self.setLabelText(self.text_mesh_out, filename)
                self.statusBar().showMessage('Ready')
            
            else:
                self.statusBar().showMessage('No output file specified!')

        else:
            self.statusBar().showMessage('No mesh data!')

    def generMesh(self):
        from seg2fem import gen_mesh_from_voxels

        etab = {1: 't', 2: 'q'}
        mtab = {1: 's', 2: 'v'}

        eid = self.rbtng_mesh_elements.checkedId()
        mid = self.rbtng_mesh_mesh.checkedId()
        self.statusBar().showMessage('Generating mesh...')
        QApplication.processEvents()
        
        self.mesh_data = gen_mesh_from_voxels(self.segmentation_data,
                                              self.voxel_sizemm * 1.0e-3,
                                              etype=etab[eid],
                                              mtype=mtab[mid])
        self.mesh_data.coors += self.dcm_offsetmm.reshape((1,3)) * 1.0e-3
        
        self.setLabelText(self.text_mesh_data, '%d %s'\
                          % (self.mesh_data.n_el,
                             elem_tab[self.mesh_data.descs[0]]))

        self.statusBar().showMessage('Ready')

    def smoothMesh(self):
        from seg2fem import gen_mesh_from_voxels_mc, smooth_mesh

        self.statusBar().showMessage('Smoothing mesh...')
        QApplication.processEvents()

        if self.mesh_smooth_method == 'marching cubes':
            self.mesh_data = gen_mesh_from_voxels_mc(self.segmentation_data,
                                                     self.voxel_sizemm * 1.0e-3)
            self.mesh_data.coors += self.dcm_offsetmm.reshape((1,3)) * 1.0e-3

        elif self.mesh_smooth_method == 'taubin':
            mid = self.rbtng_mesh_mesh.checkedId()
            if mid == 'v':
                 volume_corr=True

            else:
                volume_corr=False

            self.mesh_data.coors = smooth_mesh(self.mesh_data,
                                               n_iter=10, lam=0.6307, mu=-0.6347,
                                               volume_corr=volume_corr)
            
        self.setLabelText(self.text_mesh_data, '%d %s, smooth method - %s'\
                          % (self.mesh_data.n_el,
                             elem_tab[self.mesh_data.descs[0]],
                              self.mesh_smooth_method))

        self.statusBar().showMessage('Ready')
    
    def viewMesh(self):
        if self.mesh_data is not None:
            vtk_file = 'mesh_geom.vtk'
            self.mesh_data.write(vtk_file)
            view = QVTKViewer(vtk_file)
            view.exec_()

        else:
            self.statusBar().showMessage('No mesh data!')

    def changeOut(self, val):
        self.mesh_out_format = str(val)

    def changeSmoothMethod(self, val):
        self.mesh_smooth_method = str(val)

usage = '%prog [options]\n' + __doc__.rstrip()
help = {
    'dcm_dir': 'DICOM data direcotory',
    'dcm_file': 'DCM file with DICOM data',
    'seg_file': 'file with segmented data',
}

def main():
    parser = OptionParser(description='DICOM2FEM')
    parser.add_option('-d','--dcmdir', action='store',
                      dest='dcmdir', default=None,
                      help=help['dcm_dir'])
    parser.add_option('-m','--dcmfile', action='store',
                      dest='dcmfile', default=None,
                      help=help['dcm_file'])
    parser.add_option('-s','--segfile', action='store',
                      dest='segfile', default=None,
                      help=help['seg_file'])

    (options, args) = parser.parse_args()

    app = QApplication(sys.argv)
    mw = MainWindow(dcmdir=options.dcmdir)

    if options.dcmdir is not None:
        mw.loadDcmDir()

    if options.dcmfile is not None:
        mw.loadDcm(filename=options.dcmfile)

    if options.segfile is not None:
        mw.loadSeg(filename=options.segfile)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
