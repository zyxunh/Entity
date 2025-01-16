from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(name='CropFormer',
          author='Yixing Zhu',
          packages=find_packages(include=('CropFormer',)),
          include_package_data=True,
          )
