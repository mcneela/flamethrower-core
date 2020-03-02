from setuptools import setup

setup(
    name='flameflower',
    version='0.1',
    description='An efficient deep learning library for teaching purposes.',
    author='Daniel McNeela',
    author_email="daniel.mcneela@gmail.com",
    packages=['flameflower', 'flameflower.autograd', 'flameflower.autograd.tensor_library', 'flameflower.nn', 'flameflower.optim'],
    install_requires=['numpy>=1.12', 'future>=0.15.2'],
    keywords=['Automatic differentiation', 'backpropagation', 'gradients',
              'machine learning', 'optimization', 'neural networks',
              'Python', 'Numpy', 'Scipy', 'deep learning'],
    license=None,
)
