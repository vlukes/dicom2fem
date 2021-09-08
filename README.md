DICOM2FEM
=========

About
-----

Generation of finite element meshes using computed tomography scans.
Segmentation is based on the graph cut algorithm.

Requirements
------------

Install the required packages:

    pip install pygco scikit-learn loguru pydicom meshio imma imcut seededitorqt io3d

Installation
------------

* Download the code from the git repository:

      git clone git://github.com/vlukes/dicom2fem

or

* Use [pip](https://pypi.org/project/pip/):

      pip install dicom2fem

  or

      pip install git+git://github.com/vlukes/dicom2fem


Running the program
------------------

To run DICOM2FEM use the following command:

      python dicom2fem/dicom2fem.py

Links
-----

* https://github.com/vlukes/dicom2fem
* http://sfepy.org/dicom2fem

Author
------

Vladimir Lukeš

License
-------

MIT license
