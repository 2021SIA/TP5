﻿# Autoencoders

## Build
Requires .Net 3.1 SDK

```console
dotnet build -c Release
```

The project can also be imported in Visual Studio 2019

## Run

```console
cd TP5\bin\Release\netcoreapp3.1\
.\TP5.exe [options]
```
Information about valid arguments can be found with the --help command

```console
.\TP5.exe --help

TP5
  Simple and Multi-layer Perceptrons

Usage:
  TP5 [options]

Options:
  --config <config>  Path to the configuration file
  --version          Show version information
  -?, -h, --help     Show help and usage information
```

## Configuration

```console
.\TP5.exe --config config.yaml
```

The configuration file must have the following format:

```console
training_input: <input file path|font_set>
activation: <linear|nonlinear>
learning_rate: <learning rate>
adaptive_learning_rate: <false|true> (default=false)
momentum: <false|true> (default=false)
momentumAlpha: <alpha> (default=false)
epochs: <epochs limit>
batch: <batch_size> (default=1)
layers: <layers size array> (ex: [7,4,2,4,7])
min_error: <min error limit> (default=0)
exercise_test: <1|2> (default=1)
```



