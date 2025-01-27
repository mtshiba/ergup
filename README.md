# ergup

Install the pre-built binaries of erg. Also, build and install the package manager (poise) using the installed/existing compiler.

## requirement

Python 3

## supported platforms

* Linux x86_64, armv7, aarch64
* Windows x86_64, i686
* MacOS x86_64, aarch64

## usage

```sh
# If the Python command is registered as `python` in your environment, replace the `python3` part.
python3 <(curl -L https://github.com/mtshiba/ergup/raw/main/ergup/ergup.py)
# and please set environment variables
```

### Installing nightly version

```sh
python3 <(curl -L https://github.com/mtshiba/ergup/raw/main/ergup/ergup.py) - nightly
```

### Note

`ergup` is implemented in Erg, to solve the bootstrap problem (i.e. we can't use Erg to install Erg) we transpile the Erg script into a standalone Python script.
The following command transpiles `ergup.er` to `ergup.py`.

```sh
erg --no-std transpile ergup.er
```
