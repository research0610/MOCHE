from setuptools import setup, find_packages

setup(
    name="kstest",
    version="1.0",
    description="KS test interpreter",
    author="Zicun Cong",
    author_email="congzicun@gmail.com",
    packages=find_packages("."),
    package_data={'': ['*.json']},
    install_requires=['tqdm', 'numpy', 'TimeSynth', 'Keras == 2.2.4', 'nibabel==3.1.0', "lmdb==0.98",
                      "pyod", "xgboost", "alibi-detect", "pytz", "fbprophet", "joblib", "absl-py", "wrapt", "opt_einsum",
                      "gast", "astunparse", "termcolor", "pyyaml", "keras_applications", "keras_preprocessing",
                      "numba", "torchvision", "mlxtend", "dask", "toolz", "decorator", "dm-tree", "cloudpickle==1.3.0",
                      "creme", "fbprophet", "PyWavelets", "keras_resnet", "luminol", "statsmodels"],
    include_package_data=True,
    zip_safe=False,
)

# 'pandas==1.0.5'