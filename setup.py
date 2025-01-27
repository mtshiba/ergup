from setuptools import setup, find_packages, Command
import shutil

class Clean(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        # super().run()
        for d in ["build", "dist", "ergup.egg-info"]:
            shutil.rmtree(d, ignore_errors=True)

setup(
    name="ergup",
    author="Shunsuke Shibayama",
    author_email="sbym1346@gmail.com",
    url = "https://github.com/mtshiba/ergup",
    version="0.1.0",
    license="MIT",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'ergup=ergup.__init__:main',
        ],
    },
    cmdclass={
        'clean': Clean,
    },
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
