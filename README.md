# CS336 Spring 2026 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- [`./cs336-basics`](./cs336-basics): directory containing a module
  `cs336_basics` and its associated `pyproject.toml`. This module contains the staff 
  implementation of the language model from assignment 1. If you want to use your own 
  implementation, you can replace this directory with your own implementation.
- [`./cs336_systems`](./cs336_systems): This folder is basically empty! This is the
  module where you will implement your optimized Transformer language model. 
  Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it 
  over as a starting point. In addition, you will implement distributed training and
  optimization in this module.

Visually, it should look something like:

``` sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2 
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

0. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
package is accessible by running:

```sh
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr  7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.

## Answers to questions

### `benchmarking_script`

#### (b) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

The requested timings are as follows:

| model_size   |   warmup_steps |   measurement_steps |        forward_ms |        backward_ms |      optimizer_ms |
|:-------------|---------------:|--------------------:|------------------:|-------------------:|------------------:|
| small        |              5 |                  10 |    16.815 ± 0.129 |     32.585 ± 0.196 |    10.557 ± 0.216 |
| medium       |              5 |                  10 |    47.437 ± 0.070 |     92.925 ± 0.157 |    22.554 ± 0.264 |
| large        |              5 |                  10 |   106.996 ± 0.205 |    209.705 ± 0.523 |    40.148 ± 0.776 |
| xl           |              5 |                  10 |   297.860 ± 0.389 |    573.817 ± 0.605 |   101.431 ± 0.670 |
| 10B          |              5 |                  10 |   946.325 ± 0.146 |   1874.216 ± 1.121 |           OOM     |

The backward pass consistently takes approximately twice as long as the forward pass.  The optimization step for the largest 10B model causes a B200 GPU to run out of memory.  The variability across measurements is fairly low, and the standard deviations are small.

#### (c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

Without any warm-up steps, the requested timings are as follows:

| model_size   |   warmup_steps |   measurement_steps |         forward_ms |         backward_ms |        optimizer_ms |
|:-------------|---------------:|--------------------:|-------------------:|--------------------:|--------------------:|
| small        |              0 |                  10 |    44.634 ± 83.586 |     46.652 ± 38.114 |     11.773 ±  2.571 |
| medium       |              0 |                  10 |    51.160 ± 11.283 |     95.261 ±  6.397 |     21.773 ±  0.406 |
| large        |              0 |                  10 |   106.850 ±  0.179 |    208.883 ±  0.402 |     39.893 ±  0.553 |
| xl           |              0 |                  10 |   298.507 ±  3.021 |    572.129 ±  1.005 |    106.728 ± 19.855 |
| 10B          |              0 |                  10 |   946.147 ±  0.300 |   1872.303 ±  1.217 |             OOM     |

The standard deviations, especially for the `small` and `medium` models, are significantly higher than with warm-up steps.  This is because the first few iterations (especially the first iteration) take much longer, owing to the need to allocate memory, load kernels and modules, initialize CUDA libraries, and populate caches.  These warm-up steps are more noticeable for the `small` and `medium` models because they run first, meaning that Modal might be reusing a warm container for the larger models after the smaller models have completed.

With one warm-up step, the requested timings are as follows:

| model_size   |   warmup_steps |   measurement_steps |        forward_ms |        backward_ms |       optimizer_ms |
|:-------------|---------------:|--------------------:|------------------:|-------------------:|-------------------:|
| small        |              1 |                  10 |    20.218 ± 9.064 |     32.621 ± 0.158 |     16.692 ± 0.056 |
| medium       |              1 |                  10 |    48.533 ± 2.446 |     93.601 ± 0.442 |     32.388 ± 0.186 |
| large        |              1 |                  10 |   106.721 ± 0.071 |    209.391 ± 0.559 |     50.050 ± 0.257 |
| xl           |              1 |                  10 |   294.791 ± 0.650 |    572.191 ± 1.027 |    101.181 ± 0.254 |
| 10B          |              1 |                  10 |   945.603 ± 0.270 |   1871.444 ± 0.870 |            OOM     |

Some of the standard deviations, particularly for the `small` and `medium` model forward passes, are still higher than with five warm-up steps, though the huge initial cold-start cost is no longer present.  It is possible that CUDA, kernel, and module caches are not fully established after a single step, so there is still a small warm-up cost for the earlier measurement steps compared to the later measurement steps.