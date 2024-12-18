from setuptools import setup

setup(
    name='SpectroPy',
    version='0.1',
    description=
    'Package for computing Resonant Raman intensities, Phonon assisted Luminiscence, and exciton-phonon coupling',
    author='Muralidhar Nalabothula',
    author_email='muralidharrsvm7@gmail.com',
    packages=['SpectroPy'],
    python_requires=">=3.5",
    url='',
    license='',
    install_requires=['torch', 'numpy', 'scipy', 'numba', 'setuptools<=65.6.3'],
    scripts=[
        './exe_dips.py',
        './luminescence.py',
        './ex_ph_program.py',
        './io_exph.py',
        './exph_precision.py',
        './plot_exe_bnds.py',
        './kpts.py',
        './excitons.py',
        './exe_rep_program.py',
        './raman.py',
    ])
