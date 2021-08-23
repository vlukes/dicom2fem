import setuptools

setuptools.setup(
    name='dicom2fem',
    description='Generation of finite element meshes from DICOM images',
    long_desctioption="Generation of finite element meshes using computed " +
         "tomography scans. Segmentation is based on the graph cut algorithm.",
    version='2.0.0',
    url='https://github.com/vlukes/dicom2fem',
    author='Vladimir Lukes',
    author_email='vlukes@kme.zcu.cz',
    license='MIT',
    classifiers=[
        'Development Status :: 4',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='fem dicom',
    package_dir={"": "dicom2fem"},
    packages=setuptools.find_packages(where="dicom2fem"),
)
