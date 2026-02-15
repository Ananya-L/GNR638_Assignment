#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <omp.h>
#include <algorithm>



namespace py = pybind11;

using Tensor4D =
    std::vector<
        std::vector<
            std::vector<
                std::vector<float>
            >
        >
    >;

Tensor4D conv2d_forward(
    const Tensor4D &input,
    const Tensor4D &weight)
{
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
#pragma omp parallel for
for (int b = 0; b < B; ++b)
{
    for (int oc = 0; oc < C_out; ++oc)
    {
        for (int i = 0; i < H_out; ++i)
        {
            for (int j = 0; j < W_out; ++j)
            {
                float sum = 0.0f;

                for (int ic = 0; ic < C_in; ++ic)
                {
                    for (int ki = 0; ki < K; ++ki)
                    {
                        for (int kj = 0; kj < K; ++kj)
                        {
                            float in_val =
                                input[b][ic][i + ki][j + kj];

                            float w_val =
                                weight[oc][ic][ki][kj];

                            sum += in_val * w_val;
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


PYBIND11_MODULE(cpp_backend, m) {
    m.def("conv2d_forward", &conv2d_forward);
}
