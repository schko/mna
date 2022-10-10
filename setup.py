from setuptools import setup, find_packages, os

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='mna',
    version="1.0",
    description="Multimodal Neurophysiological Analysis",
    author="Sharath Koorathota",
    packages=find_packages(include=['mna', 'mna.*']),
    install_requires=install_requires,
    extras_require={
        'interactive': ['jupyter'],
    }
)