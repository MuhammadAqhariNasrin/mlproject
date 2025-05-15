from setuptools import find_packages, setup
from typing import List

# Constant for editable install line in requirements
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return a clean list of requirements,
    stripping newline characters and removing editable-install flags.
    '''
    with open(file_path, 'r') as file_obj:
        # Read all lines and strip whitespace/newlines
        requirements = [line.strip() for line in file_obj if line.strip()]

    # Remove the editable install directive if present
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

# Package setup
setup(
    name='mlproject',
    version='0.0.1',
    author='Aqhari',
    author_email='muhammad.aqhari.nasrin@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
  
)
