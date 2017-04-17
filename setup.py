from setuptools import setup

setup(name='parsecontent',
      version='0.1',
      description='Classify social media posts',
      url='http://github.com/dzorlu/parsecontent',
      author='Deniz Zorlu',
      author_email='dzorlu@example.com',
      license='MIT',
      packages=['lib'],
      install_requires=['tensorflow', 'keras', 'sklearn', 'numpy', 'h5py', 'pandas', 'dateutil'],
      scripts=['bin/train', 'bin/inference'],
      zip_safe=False)
