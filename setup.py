import setuptools

setuptools.setup(
    name='dicom2fem',
    description='Generation of finite element meshes from DICOM images',
    long_description="Generation of finite element meshes using computed " +
         "tomography scans. Segmentation is based on the graph cut algorithm.",
    version='2.0.0',
    url='https://github.com/vlukes/dicom2fem',
    author='Vladimir Lukes',
    author_email='vlukes@kme.zcu.cz',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords='fem dicom',
    packages=['dicom2fem'],
)
