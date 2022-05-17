# Perception monitoring and diagnosability 

## Requirements

To run this code you need:

- Python 3.8
- Poetry
- Bazel 4.2 or higher
- GCC supporting C++17 standard

The python package is managed by Poetry, while C++ is compiled by Bazel.

### Installation

To install all the python dependencies run:

```bash
poetry install --no-dev
```

Remove the `--no-dev` flag to install the development dependencies.

If you need to run the GNN, you also need to install PyTorch and PyTorch Geometric:

```bash
poetry run poe install-pytorch
```

To compile the C++ code (Percival), run:

```bash
bazel build --config c++17 ...
```

## Test

For the python package run

```bash
poetry run pytest diagnosability/tests/
```

To run C++ tests instead

```bash
bazel test --config c++17 --test_output=all ...
```

## Debug C++ Code using VSCode

First you need to compile the source code with debug symbols using the following command:

```bash
bazel run --config c++17 ... --compilation_mode=dbg
```

Them, suppose you want to debug `//percival/app:example`, edit your `launch.json` in VSCode and add

```json
{
    "name": "(gdb) Example",
    "type": "cppdbg",
    "request": "launch",
    "program": "${workspaceFolder}/bazel-bin/percival/app/example",
    "args": [],
    "stopAtEntry": false,
    "cwd": "${workspaceFolder}",
    "environment": [],
    "externalConsole": false,
    "MIMode": "gdb",
    "setupCommands": [
        {
            "description": "Enable pretty-printing for gdb",
            "text": "-enable-pretty-printing",
            "ignoreFailures": true
        }
    ]
}
```

then you can use the GDB integrated in VSCode to debug the binary.

## How to use it

Coming soon...
