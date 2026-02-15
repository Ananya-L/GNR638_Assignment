#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;

// Type aliases
using Tensor4D = std::vector<std::vector<std::vector<std::vector<float>>>>;
using Tensor2D = std::vector<std::vector<float>>;
using Tensor1D = std::vector<float>;


Tensor4D conv2d_forward(
    const Tensor4D &input,
    const Tensor4D &weight,
    const Tensor1D &bias)
{
    if (input.empty()) return {};

    int B = input.size();
    int C_in = input[0].size();
    int H = input[0][0].size();
    int W = input[0][0][0].size();

    int C_out = weight.size();
    int K = weight[0][0].size();

    int H_out = H - K + 1;
    int W_out = W - K + 1;

    Tensor4D output(
        B,
        std::vector<std::vector<std::vector<float>>>(
            C_out,
            std::vector<std::vector<float>>(
                H_out,
                std::vector<float>(W_out, 0.0f)
            )
        )
    );

    for (int b = 0; b < B; ++b)
    {
        for (int oc = 0; oc < C_out; ++oc)
        {
            for (int i = 0; i < H_out; ++i)
            {
                for (int j = 0; j < W_out; ++j)
                {
                    float sum = bias[oc];

                    for (int ic = 0; ic < C_in; ++ic)
                    {
                        for (int ki = 0; ki < K; ++ki)
                        {
                            for (int kj = 0; kj < K; ++kj)
                            {
                                sum += input[b][ic][i + ki][j + kj] *
                                       weight[oc][ic][ki][kj];
                            }
                        }
                    }

                    output[b][oc][i][j] = sum;
                }
            }
        }
    }

    return output;
}


// =============================
// MaxPool2D forward
// =============================
Tensor4D maxpool2d_forward(const Tensor4D &input, int kernel_size)
{
    if (input.empty()) return {};

    int B = input.size();
    int C = input[0].size();
    int H = input[0][0].size();
    int W = input[0][0][0].size();

    int H_out = H / kernel_size;
    int W_out = W / kernel_size;

    Tensor4D output(
        B,
        std::vector<std::vector<std::vector<float>>>(
            C,
            std::vector<std::vector<float>>(
                H_out,
                std::vector<float>(W_out, 0.0f)
            )
        )
    );

    for (int b = 0; b < B; ++b)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int i = 0; i < H_out; ++i)
            {
                for (int j = 0; j < W_out; ++j)
                {
                    float max_val = input[b][c][i * kernel_size][j * kernel_size];

                    for (int ki = 0; ki < kernel_size; ++ki)
                    {
                        for (int kj = 0; kj < kernel_size; ++kj)
                        {
                            float val = input[b][c][i * kernel_size + ki][j * kernel_size + kj];
                            if (val > max_val)
                                max_val = val;
                        }
                    }

                    output[b][c][i][j] = max_val;
                }
            }
        }
    }

    return output;
}


// =============================
// ReLU forward
// =============================
Tensor4D relu_forward(const Tensor4D &input)
{
    if (input.empty()) return {};

    Tensor4D output = input;

    int B = output.size();
    int C = output[0].size();
    int H = output[0][0].size();
    int W = output[0][0][0].size();

    for (int b = 0; b < B; ++b)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int h = 0; h < H; ++h)
            {
                for (int w = 0; w < W; ++w)
                {
                    if (output[b][c][h][w] < 0.0f)
                        output[b][c][h][w] = 0.0f;
                }
            }
        }
    }

    return output;
}


// =============================
// Flatten forward
// =============================
Tensor2D flatten_forward(const Tensor4D &input)
{
    if (input.empty()) return {};

    int B = input.size();
    Tensor2D output(B);

    for (int b = 0; b < B; ++b)
    {
        std::vector<float> flat;

        for (size_t c = 0; c < input[b].size(); ++c)
        {
            for (size_t h = 0; h < input[b][c].size(); ++h)
            {
                for (size_t w = 0; w < input[b][c][h].size(); ++w)
                {
                    flat.push_back(input[b][c][h][w]);
                }
            }
        }

        output[b] = flat;
    }

    return output;
}


// =============================
// MatMul forward (Fixed bias shape)
// =============================
Tensor2D matmul_forward(
    const Tensor2D &input,
    const Tensor2D &weight,
    const Tensor2D &bias)   // <-- FIXED
{
    if (input.empty()) return {};

    int B = input.size();
    int in_feat = input[0].size();
    int out_feat = weight[0].size();

    Tensor2D output(B, std::vector<float>(out_feat, 0.0f));

    for (int b = 0; b < B; ++b)
    {
        for (int j = 0; j < out_feat; ++j)
        {
            float sum = bias[0][j];   // <-- FIXED

            for (int i = 0; i < in_feat; ++i)
            {
                sum += input[b][i] * weight[i][j];
            }

            output[b][j] = sum;
        }
    }

    return output;
}


PYBIND11_MODULE(cpp_backend, m)
{
    m.doc() = "Optimized C++ backend for neural network operations";

    m.def("conv2d_forward", &conv2d_forward);
    m.def("maxpool2d_forward", &maxpool2d_forward);
    m.def("relu_forward", &relu_forward);
    m.def("flatten_forward", &flatten_forward);
    m.def("matmul_forward", &matmul_forward);
}
