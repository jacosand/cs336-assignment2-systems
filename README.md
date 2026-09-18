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

### `nys_profile`

#### (a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

| model_size | context_length | forward_ms |
|:-----------|---------------:|-----------:|
| small      |  256           |  26.338    |
| small      |  512           |  23.792    |
| small      | 1024           |  38.779    |
| xl         |  256           | 158.757    |
| xl         |  512           | 297.804    |
| xl         | 1024           | 587.743    |

For the `xl` model, the timing for `context_length = 512` is quite close, but for the `small` model, the timing here is longer, likely because of the overhead of running the profiler itself.

#### (b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes?

For the forward pass, we have:

| model_size | context_length | longest_kernel                                                                               | times_invoked |
|:-----------|---------------:|---------------------------------------------------------------------------------------------:|--------------:|
| small      |  256           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`   |  24           |
| small      |  512           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x128x16_1x1x1_3_tnn_align1_bias_f32_relu`  |  25           |
| small      | 1024           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_tnn_align1_bias_f32_relu`   |  60           |
| xl         |  256           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_tnn_align1_bias_f32_relu`  |  97           |
| xl         |  512           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_tnn_align1_bias_f32_relu` |  65           |
| xl         | 1024           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_tnn_align1_bias_f32_relu`  | 160           |

For the forward and backward pass, we have:

| model_size | context_length | longest_kernel                                                                               | times_invoked |
|:-----------|---------------:|---------------------------------------------------------------------------------------------:|--------------:|
| small      |  256           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_nnn_align1_bias_f32_relu`  |  72           |
| small      |  512           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu`   | 109           |
| small      | 1024           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu`   |  73           |
| xl         |  256           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x128x16_1x1x1_3_ntn_align1_bias_f32_relu` |  97           |
| xl         |  512           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x64x16_1x1x1_3_nnn_align1_bias_f32_relu`   | 193           |
| xl         | 1024           | `cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_128x64x16_1x1x1_3_nnn_align1_bias_f32_relu`  | 193           |

All of these are matrix multiplication kernels, but they are for matrices of different sizes.  For some model sizes and context lengths, the kernel that takes the most GPU time for forward-only and forward-and-backward passes are the same, but for others, the kernel is different.

#### (c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

Some other kernels besides matrix multiplies that account for non-trivial CUDA runtime in the forward pass are:

- `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::DivFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`
- `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::<unnamed>::where_kernel_impl(at::TensorIterator &)::[lambda() (instance 1)]::operator ()() const::[lambda() (instance 11)]::operator ()() const::[lambda(bool, float, float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`
- `void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)`

These are elementwise operations: the first is for elementwise divsion, the second is for `torch.where` used in masking, and the third is for elementwise addition.  These elementwise kernels account for a larger percentage of the runtime in the `small` model compared to the `xl` model.

#### (d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you'd do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

For the forward pass, we have:

| model_size | context_length | matrix_multiplication_fraction |
|:-----------|---------------:|-------------------------------:|
| small      |  256           |  0.772                         |
| small      |  512           |  0.759                         |
| small      | 1024           |  0.689                         |
| xl         |  256           |  0.930                         |
| xl         |  512           |  0.906                         |
| xl         | 1024           |  0.862                         |

For the full training step, we have:

| model_size | context_length | matrix_multiplication_fraction |
|:-----------|---------------:|-------------------------------:|
| small      |  256           |  0.581                         |
| small      |  512           |  0.642                         |
| small      | 1024           |  0.629                         |
| xl         |  256           |  0.744                         |
| xl         |  512           |  0.803                         |
| xl         | 1024           |  0.812                         |

The fraction of time spent on matrix multiplication decreases for the full training step compared to the forward pass, at least in part because the AdamW optimization step consists almost entirely of elementwise operations.  Note also that the fraction of matrix multiplication is higher for a larger model size.

#### (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

We have:

| model_size | context_length | matrix_multiplication_mean_µs | softmax_mean_µs |
|:-----------|---------------:|------------------------------:|----------------:|
| small      |  256           |    72.828                     |   80.900        |
| small      |  512           |   131.359                     |  133.680        |
| small      | 1024           |   428.450                     |  479.059        |
| xl         |  256           |   108.286                     |   94.778        |
| xl         |  512           |   361.932                     |  340.982        |
| xl         | 1024           |  1303.744                     | 1254            |

The runtimes are comparable for matrix multiplication and softmax, but the number of operations in each layer for matrix multiplication is about `4 * context_length * context_length * d_model` while the number of operations in each layer for softmax is about `5 * context_length * context_length * num_heads`.  Thus, the ratio of FLOPS for matrix multiplication to FLOPs for softmax is `4 * d_model / (5 * num_heads)` which is much greater than 1.  (For the `small` model the ratio is about 51.2 and for the `xl` model the ratio is about 64.)

The reason for comparable runtimes despite many more matrix multiplication FLOPs is that matrix multiplication has much higher arithmetic intensity through data reuse; thus it is much more efficient on the GPU.  Softmax is less efficient, involving the launching of many different kernels and more frequent reading and writing from memory.