# ergup

## requirement

Python 3.10+

## support platform

* Linux x86_64
* Windows x86_64

## usage

```console
curl -O -L https://github.com/mtshiba/ergup/raw/main/bin/ergup`python -c "import sys; print('%d%d' % (sys.version_info.major, sys.version_info.minor))"`.pyc || echo not supported python version
python3 ergup`python -c "import sys; print('%d%d' % (sys.version_info.major, sys.version_info.minor))"`.pyc

# please set envs
export PATH=$PATH:.erg/bin
export ERG_PATH=.erg
```

That's all!
