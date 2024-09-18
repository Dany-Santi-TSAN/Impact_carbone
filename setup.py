from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
   name='impact_carbone',
   version='0.0.1',
   description='Une app qui te permet de connaitre ton réel impact carbone',
   #long_description=__doc__,
   url='https://github.com/Dany-Santi-TSAN/Impact_carbone',
   author='Dany TSAN',
   author_email='danytsan23@gmail.com',
   license='MIT',
   install_requires=requirements,
   packages=find_packages(),
   )
