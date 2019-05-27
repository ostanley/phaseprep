from distutils.core import setup

setup(
    # Project information
    name='phaseprep',
    description='Pipelines related to phase regression',
    packages=['phaseprep',
              'phaseprep/interfaces',
              'phaseprep/workflows'],

    # Metadata
    author='Olivia Stanley',
    author_email='ostanle2@uwo.ca',
    url='https://git.cfmm.robarts.ca/ostanle2/phaseprep'
)
