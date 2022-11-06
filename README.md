# ergup

## requirement

Python 3

## supported platforms

* Linux x86_64
* Windows x86_64
* (MacOS) x86_64, aarch64 (not tested)

## usage

```console
filename=ergup`python -c "import sys; print('%d%d' % (sys.version_info.major, sys.version_info.minor))"`.pyc
curl -O -L https://github.com/mtshiba/ergup/raw/main/bin/$filename
grep -q "404" $filename && echo "not supported python version" || python3 $filename

# please set envs
export PATH=$PATH:.erg/bin
export ERG_PATH=.erg
```

That's all!
