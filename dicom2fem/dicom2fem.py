#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DICOM2FEM - organ segmentation and FE model generator

Example:

$ dicom2fem -d sample_data
"""
#TODO:
# recalculate voxelsize when rescaled

# import unittest
from optparse import OptionParser
from scipy.io import loadmat, savemat
from scipy import ndimage
import numpy as np
import sys
import os

from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
     QHBoxLayout, QVBoxLayout, QTabWidget,\
     QLabel, QPushButton, QFrame, QFileDialog,\
     QFont, QInputDialog, QComboBox, QPixmap
from PyQt4.Qt import QString

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', 'pyseg_base', 'pysegbase'))

import dcmreaddata as dcmreader
from seed_editor_qt import QTSeedEditor
import pycut

from meshio import supported_capabilities, supported_formats, MeshIO
from seg2fem import gen_mesh_from_voxels, gen_mesh_from_voxels_mc
from seg2fem import smooth_mesh

from viewer import QVTKViewer

inv_supported_formats = dict(zip(supported_formats.values(),
                                 supported_formats.keys()))
smooth_methods = {
    'taubin vol.': (smooth_mesh, {'n_iter': 10, 'volume_corr': True,
                             'lam': 0.6307, 'mu': -0.6347}),
    'taubin': (smooth_mesh, {'n_iter': 10, 'volume_corr': False,
                             'lam': 0.6307, 'mu': -0.6347}),

    }

mesh_generators = {
    'surface/tri': (6, gen_mesh_from_voxels, {'etype': 't', 'mtype': 's'}),
    'surface/quad': (5, gen_mesh_from_voxels, {'etype': 'q', 'mtype': 's'}),
    'volume/tetra': (4, gen_mesh_from_voxels, {'etype': 't', 'mtype': 'v'}),
    'volume/hexa': (3, gen_mesh_from_voxels, {'etype': 'q', 'mtype': 'v'}),
    'march. cubes - surf.': (2, gen_mesh_from_voxels_mc, {}),
    'march. cubes - vol.': (1, gen_mesh_from_voxels_mc, {'gmsh3d': True}),
    }

elem_tab = {
    '2_3': 'triangles',
    '3_4': 'tetrahedrons',
    '2_4': 'quads',
    '3_8': 'hexahedrons'
    }

class MainWindow(QMainWindow):

    def __init__(self, dcmdir=None):
        QMainWindow.__init__(self)

        self.dcmdir = dcmdir
        self.dcm_3Ddata = None
        self.dcm_metadata = None
        self.dcm_zoom = np.array([1.0, 1.0, 1.0])
        self.dcm_offsetmm = np.array([0,0,0])
        self.voxel_volume = 0.0
        self.voxel_sizemm = None
        self.voxel_sizemm_scaled = None
        self.segmentation_seeds = None
        self.segmentation_data = None
        self.segmentation_data_scaled = None
        self.mesh_data = None
        self.mesh_out_format = 'vtk'
        self.mesh_smooth_method = 'taubin vol.'
        self.initUI()

    def init_ReaderTab(self):
        vbox = QVBoxLayout()
        vbox.setSpacing(10)

        self.text_dcm_dir = QLabel('DICOM dir:')
        self.text_dcm_data = QLabel('DICOM data:')
        vbox.addWidget(QLabel())
        vbox.addWidget(self.text_dcm_dir)
        vbox.addWidget(self.text_dcm_data)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        vbox.addWidget(hr)

        vbox1 = QVBoxLayout()
        btn_dcmdir = QPushButton("Load DICOM", self)
        btn_dcmdir.clicked.connect(self.loadDcmDir)
        btn_dcmsave = QPushButton("Save DCM", self)
        btn_dcmsave.clicked.connect(self.saveDcm)
        vbox1.addWidget(btn_dcmdir)
        vbox1.addWidget(btn_dcmsave)
        vbox1.addStretch(1)

        vbox2 = QVBoxLayout()
        btn_dcmred = QPushButton("Rescale", self)
        btn_dcmred.clicked.connect(self.rescaleDcm)
        btn_dcmcrop = QPushButton("Crop", self)
        btn_dcmcrop.clicked.connect(self.cropDcm)
        vbox2.addWidget(btn_dcmred)
        vbox2.addWidget(btn_dcmcrop)
        vbox2.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox1)
        hbox.addStretch(1)
        hbox.addLayout(vbox2)
        hbox.addStretch(1)

        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)

        return vbox

    def init_SegmentationTab(self):
        vbox = QVBoxLayout()
        vbox.setSpacing(10)

        self.text_seg_in = QLabel('input data:')
        self.text_seg_data = QLabel('segment. data:')
        vbox.addWidget(QLabel())
        vbox.addWidget(self.text_seg_in)
        vbox.addWidget(self.text_seg_data)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        vbox.addWidget(hr)

        vbox1 = QVBoxLayout()
        btn_segload = QPushButton("Load DCM", self)
        btn_segload.clicked.connect(self.loadDcm)
        btn_segsave = QPushButton("Save SEG", self)
        btn_segsave.clicked.connect(self.saveSeg)
        vbox1.addWidget(btn_segload)
        vbox1.addWidget(btn_segsave)
        vbox1.addStretch(1)

        vbox2 = QVBoxLayout()
        btn_maskreg = QPushButton("Mask region", self)
        btn_maskreg.clicked.connect(self.maskRegion)
        btn_segauto = QPushButton("Automatic seg.", self)
        btn_segauto.clicked.connect(self.autoSeg)
        btn_segman = QPushButton("Manual seg.", self)
        btn_segman.clicked.connect(self.manualSeg)
        vbox2.addWidget(btn_maskreg)
        vbox2.addWidget(btn_segauto)
        vbox2.addWidget(btn_segman)
        vbox2.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox1)
        hbox.addStretch(1)
        hbox.addLayout(vbox2)
        hbox.addStretch(1)

        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)

        return vbox

    def init_MeshGenTab(self):
        vbox = QVBoxLayout()
        vbox.setSpacing(10)

        self.text_mesh_in = QLabel('input data:')
        self.text_mesh_data = QLabel('mesh data:')
        vbox.addWidget(QLabel())
        vbox.addWidget(self.text_mesh_in)
        vbox.addWidget(self.text_mesh_data)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        vbox.addWidget(hr)

        self.text_mesh_grid = QLabel('grid:')
        vbox.addWidget(self.text_mesh_grid)
        btn_meshrescale = QPushButton("Rescale", self)
        btn_meshrescale.clicked.connect(self.rescaleSeg)
        hbox0 = QHBoxLayout()
        hbox0.addStretch(1)
        hbox0.addWidget(btn_meshrescale)
        hbox0.addStretch(1)
        vbox.addLayout(hbox0)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        vbox.addWidget(hr)

        vbox1 = QVBoxLayout()
        btn_meshload = QPushButton("Load SEG", self)
        btn_meshload.clicked.connect(self.loadSeg)
        btn_meshsave = QPushButton("Save MESH", self)
        btn_meshsave.clicked.connect(self.saveMesh)
        text_mesh_output = QLabel('format:')

        combo_sm = QComboBox(self)
        combo_sm.activated[str].connect(self.changeOut)
        supp_write = [k for k, v in supported_capabilities.iteritems()\
                      if 'w' in v]

        supp_write.sort()
        combo_sm.addItems(supp_write)
        combo_sm.setCurrentIndex(supp_write.index('vtk'))

        vbox1.addWidget(btn_meshload)
        vbox1.addWidget(btn_meshsave)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(text_mesh_output)
        hbox1.addWidget(combo_sm)
        vbox1.addLayout(hbox1)
        vbox1.addStretch(1)

        vbox2 = QVBoxLayout()
        btn_meshgener = QPushButton("Generate", self)
        btn_meshgener.clicked.connect(self.generMesh)
        text_mesh_mesh = QLabel('generator:')

        combo_mg = QComboBox(self)
        combo_mg.activated[str].connect(self.changeMesh)
        mg_labels = [(k, v[0]) for k, v in mesh_generators.iteritems()]
        mg_labels.sort(key=lambda tup: tup[1])
        mg_items = [ii[0] for ii in mg_labels]
        self.mesh_generator = mg_items[0]
        combo_mg.addItems(mg_items)
        combo_mg.setCurrentIndex(mg_items.index(self.mesh_generator))

        vbox2.addWidget(btn_meshgener)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(text_mesh_mesh)
        hbox1.addWidget(combo_mg)
        vbox2.addLayout(hbox1)
        #vbox2.addStretch(1)
        vbox2.addWidget(QLabel())

        btn_meshsmooth = QPushButton("Smooth", self)
        btn_meshsmooth.clicked.connect(self.smoothMesh)
        text_mesh_smooth = QLabel('method:')

        combo_out = QComboBox(self)
        combo_out.activated[str].connect(self.changeSmoothMethod)
        keys = smooth_methods.keys()
        combo_out.addItems(keys)
        combo_out.setCurrentIndex(keys.index('taubin vol.'))

        vbox2.addWidget(btn_meshsmooth)
        hbox1 = QHBoxLayout()
        hbox1.addWidget(text_mesh_smooth)
        hbox1.addWidget(combo_out)
        vbox2.addLayout(hbox1)
        vbox2.addStretch(1)

        vbox3 = QVBoxLayout()
        btn_meshview = QPushButton("Mesh preview", self)
        btn_meshview.clicked.connect(self.viewMesh)
        vbox3.addWidget(btn_meshview)
        vbox3.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox1)
        hbox.addStretch(1)
        hbox.addLayout(vbox2)
        hbox.addStretch(1)
        hbox.addLayout(vbox3)
        hbox.addStretch(1)

        vbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addStretch(1)

        return vbox

    def initUI(self):
        import os.path as op
        path_to_script = op.dirname(os.path.abspath(__file__))

        cw = QWidget()
        self.setCentralWidget(cw)
        vbox = QVBoxLayout()
        vbox.setSpacing(10)

        # status bar
        self.statusBar().showMessage('Ready')

        # info panel
        font_label = QFont()
        font_label.setBold(True)
        font_info = QFont()
        font_info.setItalic(True)
        font_info.setPixelSize(10)

        dicom2fem_title = QLabel('DICOM2FEM')
        info = QLabel('Version: 0.91\n\n' +
                      'Developed by:\n' +
                      'University of West Bohemia\n' +
                      'Faculty of Applied Sciences\n' +
                      QString.fromUtf8('V. Lukeš - 2015') +
                      '\n\nBased on PYSEG_BASE project'
                      )
        info.setFont(font_info)
        dicom2fem_title.setFont(font_label)
        dicom2fem_logo = QLabel()
        logopath = os.path.join(path_to_script, 'brain.png')
        logo = QPixmap(logopath)
        dicom2fem_logo.setPixmap(logo)
        vbox1 = QVBoxLayout()
        vbox1.addWidget(dicom2fem_title)
        vbox1.addWidget(info)
        vbox1.addStretch(1)

        vbox2 = QVBoxLayout()
        vbox2.addWidget(dicom2fem_logo)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox1)
        hbox.addStretch(1)
        hbox.addLayout(vbox2)
        hbox.addStretch(1)
        vbox.addLayout(hbox)

        tabs = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()

        tab1.setLayout(self.init_ReaderTab())
        tab2.setLayout(self.init_SegmentationTab())
        tab3.setLayout(self.init_MeshGenTab())
        tabs.addTab(tab1,"DICOM Reader")
        tabs.addTab(tab2,"Segmentation")
        tabs.addTab(tab3,"Mesh generator")

        vbox.addWidget(tabs)

        # clear, quit
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        btn_clear = QPushButton("Clear", self)
        btn_clear.clicked.connect(self.clearall)
        hbox.addWidget(btn_clear)
        btn_quit = QPushButton("Quit", self)
        btn_quit.clicked.connect(self.quit)
        hbox.addWidget(btn_quit)
        hbox.addStretch(1)

        vbox.addLayout(hbox)

        cw.setLayout(vbox)
        self.setWindowTitle('DICOM2FEM')
        self.show()

    def quit(self, event):
        self.close()

    def clearall(self, event):
        self.dcmdir = None
        del(self.dcm_3Ddata)
        self.dcm_3Ddata= None
        del(self.dcm_metadata)
        self.dcm_metadata = None
        self.dcm_zoom = np.array([1.0, 1.0, 1.0])
        self.dcm_offsetmm = np.array([0,0,0])
        self.voxel_volume = 0.0
        self.voxel_sizemm = None
        self.voxel_sizemm_scaled = None
        del(self.segmentation_seeds)
        self.segmentation_seeds = None
        del(self.segmentation_data)
        self.segmentation_data = None
        del(self.segmentation_data_scaled)
        self.segmentation_data_scaled = None
        del(self.mesh_data)
        self.mesh_data = None


        self.setLabelText(self.text_dcm_dir, '')
        self.setLabelText(self.text_dcm_data, '')
        self.setLabelText(self.text_seg_in, '')
        self.setLabelText(self.text_seg_data, '')
        self.setLabelText(self.text_mesh_in, '')
        self.setLabelText(self.text_mesh_data, '')
        self.setLabelText(self.text_mesh_grid, '')

    def setLabelText(self, obj, text):
        dlab = str(obj.text())
        obj.setText(dlab[:dlab.find(':')] + ': %s' % text)

    @staticmethod
    def getSizeInfo(voxelsize, data):
        vsize = tuple([float(ii) for ii in voxelsize])
        ret = ' %dx%dx%d,  %fx%fx%f mm' % (data.shape + vsize)

        return ret

    def getDcmInfo(self):
        return self.getSizeInfo(self.voxel_sizemm, self.dcm_3Ddata)

    def getSegInfo(self):
        if self.segmentation_data_scaled is not None:
            segdata = self.segmentation_data_scaled
            voxelsize = self.voxel_sizemm_scaled

        else:
            segdata = self.segmentation_data
            voxelsize = self.voxel_sizemm

        return self.getSizeInfo(voxelsize, segdata)

    def setVoxelVolume(self, vxs):
        self.voxel_volume = np.prod(vxs)

    def loadDcmDir(self):
        self.statusBar().showMessage('Reading DICOM directory...')
        QApplication.processEvents()

        if self.dcmdir is None:
            self.dcmdir = dcmreader.get_dcmdir_qt(app=True)

        if self.dcmdir is not None:
            dcr = dcmreader.DicomReader(os.path.abspath(self.dcmdir),
                                        qt_app=self)
        else:
            self.statusBar().showMessage('No DICOM directory specified!')
            return

        if dcr.validData():
            self.dcm_3Ddata = dcr.get_3Ddata()
            self.dcm_metadata = dcr.get_metaData()
            self.voxel_sizemm = np.array(self.dcm_metadata['voxelsize_mm'])
            self.setVoxelVolume(self.voxel_sizemm)
            self.setLabelText(self.text_dcm_dir, self.dcmdir)
            self.setLabelText(self.text_dcm_data, self.getDcmInfo())
            self.statusBar().showMessage('Ready')
            self.setLabelText(self.text_seg_in, 'DICOM reader')

        else:
            self.statusBar().showMessage('No DICOM data in direcotry!')

    def getRescaleValues(self, new_vsize, old_vsize, labels):
        aux = '%.2f,%.2f,%.2f' % tuple(new_vsize)
        value, ok = QInputDialog.getText(self, labels[0], labels[1],
                                         text=aux)
        if ok and value is not None:
            vals = value.split(',')
            if len(vals) == 3:
                new_vsize = [float(ii) for ii in vals]

            else:
                aux = float(vals[0])
                new_vsize = [aux, aux, aux]

        zoom = old_vsize / np.array(new_vsize)

        for ii in zoom:
           if ii < 0.0 or ii > 100:
               return None

        return zoom

    def rescaleDcm(self, event=None, new_vsize=[1.0,1.0,1.0]):
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Rescaling DICOM data...')
        QApplication.processEvents()

        if event is not None:
            zoom = self.getRescaleValues(new_vsize, self.voxel_sizemm,
                                         ('Rescale DICOM data',
                                          'Voxel size [mm]:'))

        if zoom is not None:
            self.dcm_zoom *= zoom
            self.dcm_3Ddata = ndimage.zoom(self.dcm_3Ddata, zoom,
                                           prefilter=False, mode='nearest')
            self.voxel_sizemm /= zoom
            self.setLabelText(self.text_dcm_data, self.getDcmInfo())

            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('Invalid voxel size!')

    def cropDcm(self):
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Cropping DICOM data...')
        QApplication.processEvents()

        pyed = QTSeedEditor(self.dcm_3Ddata, mode='crop',
                            voxelSize=self.voxel_sizemm)
        pyed.exec_()
        self.dcm_3Ddata = pyed.getImg()
        self.dcm_offsetmm = pyed.getOffset()

        self.setLabelText(self.text_dcm_data, self.getDcmInfo())
        self.statusBar().showMessage('Ready')

        self.statusBar().showMessage('Ready')

    def saveDcm(self, event=None, filename=None):
        if self.dcm_3Ddata is not None:
            self.statusBar().showMessage('Saving DICOM data...')
            QApplication.processEvents()

            if filename is None:
                filename = \
                    str(QFileDialog.getSaveFileName(self,
                                                    'Save DCM file',
                                                    filter='Files (*.dcm)'))
            if len(filename) > 0:
                savemat(filename, {'data': self.dcm_3Ddata,
                                   'voxelsize_mm': self.voxel_sizemm,
                                   'offset_mm': self.dcm_offsetmm},
                                   appendmat=False)

                #self.setLabelText(self.text_dcm_out, filename)
                self.statusBar().showMessage('Ready')

            else:
                self.statusBar().showMessage('No output file specified!')

        else:
            self.statusBar().showMessage('No DICOM data!')

    def loadDcm(self, event=None, filename=None):
        self.statusBar().showMessage('Loading DICOM data...')
        QApplication.processEvents()

        if filename is None:
            filename = str(QFileDialog.getOpenFileName(self, 'Load DCM file',
                                                       filter='Files (*.dcm)'))

        if len(filename) > 0:

            data = loadmat(filename,
                           variable_names=['data', 'voxelsize_mm', 'offset_mm'],
                           appendmat=False)

            self.dcm_3Ddata = data['data']
            self.voxel_sizemm = data['voxelsize_mm'].reshape((3,))
            self.dcm_offsetmm = data['offset_mm'].reshape((3,))
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
            self.setLabelText(self.text_mesh_grid, self.getSegInfo())
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('Zero SEG data!')

    def maskRegion(self):
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Mask region...')
        QApplication.processEvents()

        pyed = QTSeedEditor(self.dcm_3Ddata, mode='mask',
                            voxelSize=self.voxel_sizemm)

        pyed.exec_()

        self.statusBar().showMessage('Ready')

    def autoSeg(self):
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        igc = pycut.ImageGraphCut(self.dcm_3Ddata,
                                  voxelsize=self.voxel_sizemm)

        pyed = QTSeedEditor(self.dcm_3Ddata,
                            seeds=self.segmentation_seeds,
                            modeFun=igc.interactivity_loop,
                            voxelSize=self.voxel_sizemm)
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
                            voxelSize=self.voxel_sizemm)
        pyed.exec_()

        self.segmentation_data = pyed.getSeeds()
        self.checkSegData()

    def saveSeg(self, event=None, filename=None):
        if self.segmentation_data is not None:
            self.statusBar().showMessage('Saving segmentation data...')
            QApplication.processEvents()

            if filename is None:
                filename = \
                    str(QFileDialog.getSaveFileName(self,
                                                    'Save SEG file',
                                                    filter='Files (*.seg)'))

            if len(filename) > 0:

                outdata = {'data': self.dcm_3Ddata,
                           'segdata': self.segmentation_data,
                           'voxelsize_mm': self.voxel_sizemm,
                           'offset_mm': self.dcm_offsetmm}

                if self.segmentation_seeds is not None:
                    outdata['segseeds'] = self.segmentation_seeds

                savemat(filename, outdata, appendmat=False)
                #self.setLabelText(self.text_seg_out, filename)
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
            QApplication.processEvents()

            data = loadmat(filename,
                           variable_names=['data', 'segdata', 'segseeds',
                                           'voxelsize_mm', 'offset_mm'],
                           appendmat=False)

            self.dcm_3Ddata = data['data']
            self.segmentation_data = data['segdata']
            if 'segseeds' in data:
                self.segmentation_seeds = data['segseeds']

            else:
                self.segmentation_seeds = None

            self.voxel_sizemm = data['voxelsize_mm'].reshape((3,))
            self.dcm_offsetmm = data['offset_mm'].reshape((3,))
            self.setVoxelVolume(self.voxel_sizemm.reshape((3,)))
            self.setLabelText(self.text_mesh_in, filename)
            self.setLabelText(self.text_mesh_grid, self.getSegInfo())
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No input file specified!')

    def rescaleSeg(self, event=None, new_vsize=[1.0,1.0,1.0]):
        if self.segmentation_data is None:
            self.statusBar().showMessage('No segmentation data!')
            return

        self.statusBar().showMessage('Rescaling segmentation data...')
        QApplication.processEvents()

        zoom = self.getRescaleValues(new_vsize, self.voxel_sizemm,
                                     ('Rescale segmentation data',
                                      'Grid size [mm]:'))

        if zoom is not None:
            self.dcm_zoom *= zoom
            self.segmentation_data_scaled =\
                ndimage.zoom(self.segmentation_data, zoom,
                             prefilter=False, mode='nearest')
            self.voxel_sizemm_scaled = self.voxel_sizemm / zoom
            self.setLabelText(self.text_mesh_grid, self.getSegInfo())

            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('Invalid grid size!')

    def saveMesh(self, event=None, filename=None):
        if self.mesh_data is not None:
            self.statusBar().showMessage('Saving mesh...')
            QApplication.processEvents()

            if filename is None:
                file_ext = inv_supported_formats[self.mesh_out_format]
                filename = \
                    str(QFileDialog.getSaveFileName(self, 'Save MESH file',
                                                    filter='Files (*%s)'\
                                                        % file_ext))

            if len(filename) > 0:

                io = MeshIO.for_format(filename, format=self.mesh_out_format,
                                       writable=True)
                io.write(filename, self.mesh_data)

                self.statusBar().showMessage('Ready')

            else:
                self.statusBar().showMessage('No output file specified!')

        else:
            self.statusBar().showMessage('No mesh data!')

    def generMesh(self):
        def getScaleFactor(self, value=0.25):
            label = ('Marching cubes - volume remeshing',
                     'Characteristic Length Factor:')
            value, ok = QInputDialog.getText(self, label[0], label[1],
                                             text='%.2f' % value)
            if ok and value is not None:
                value = float(value)

            if value > 0. and value < 10.:
                return value

            else:
                return None

        self.statusBar().showMessage('Generating mesh...')
        QApplication.processEvents()

        if self.segmentation_data_scaled is not None:
            segdata = self.segmentation_data_scaled
            voxelsize = self.voxel_sizemm_scaled * 1.0e-3

        else:
            segdata = self.segmentation_data
            voxelsize = self.voxel_sizemm * 1.0e-3

        if segdata is not None:
            mgid, gen_fun, pars = mesh_generators[self.mesh_generator]
            if mgid == 1:
                pars['scale_factor'] = getScaleFactor(self, value=0.25)

            self.mesh_data = gen_fun(segdata, voxelsize, **pars)

            self.mesh_data.coors += self.dcm_offsetmm * 1.0e-3

            self.setLabelText(self.text_mesh_data, '%d %s'\
                                  % (self.mesh_data.n_el,
                                     elem_tab[self.mesh_data.descs[0]]))

            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No segmentation data!')

    def smoothMesh(self):
        self.statusBar().showMessage('Smoothing mesh...')
        QApplication.processEvents()

        if self.mesh_data is not None:
            smooth_fun, pars = smooth_methods[self.mesh_smooth_method]
            etype = '%d_%d' % (self.mesh_data.dim,
                               self.mesh_data.conns[0].shape[-1])
            if (etype == '2_2' or etype == '3_3') and pars['volume_corr']:
                self.statusBar().showMessage('No volume mesh!')

            else:
                self.mesh_data.coors = smooth_fun(self.mesh_data, **pars)

                self.setLabelText(self.text_mesh_data,
                                  '%d %s, smooth method - %s'\
                                      % (self.mesh_data.n_el,
                                         elem_tab[self.mesh_data.descs[0]],
                                         self.mesh_smooth_method))

                self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No mesh data!')

    def viewMesh(self):
        if self.mesh_data is not None:
            vtk_file = 'mesh_geom.vtk'
            self.mesh_data.write(vtk_file)
            view = QVTKViewer(vtk_file)
            view.exec_()

        else:
            self.statusBar().showMessage('No mesh data!')

    def changeMesh(self, val):
        self.mesh_generator = str(val)

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
    parser.add_option('-f','--dcmfile', action='store',
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
