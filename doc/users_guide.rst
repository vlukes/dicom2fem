Users guide
===========

Installation
------------

Install DICOM2FEM::

    git clone --recursive https://github.com/vlukes/dicom2fem.git


Follow the instructions in pyseg_base/INSTALL.


Sample DICOM data can be found at:

  * http://www.mathworks.com/matlabcentral/fileexchange/2762-dicom-example-files?download=true
  * http://www.osirix-viewer.com/datasets/


Running the program
-------------------

Command line usage::

    $ python src/dicom2fem.py --help
    Usage: dicom2fem.py [options]

    Options:

      -h, --help                      show this help message and exit
      -d DCMDIR, --dcmdir=DCMDIR      DICOM data direcotory
      -m DCMFILE, --dcmfile=DCMFILE   DCM file with DICOM data
      -s SEGFILE, --segfile=SEGFILE   file with segmented data
