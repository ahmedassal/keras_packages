from distutils.core import setup


def readme():
  with open('README.md') as f:
    return f.read()

setup(
  name='DataGenerators',
  version='0.1.0',
  packages=['DataGenerators.DataGenerators'],
  url='https://github.com/ahmedassal/DataGenerators',
  license='MIT',
  author='Ahmed Assal',
  author_email='ahmed.assal@vertex-techs.com',
  description='DataGenerators'
)
