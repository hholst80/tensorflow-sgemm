#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <iostream>
#include <cblas.h>
#include <cstdio>

using namespace tensorflow;

Status MatMulShape(shape_inference::InferenceContext* c) {
  using namespace shape_inference;
  ShapeHandle a;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));

  ShapeHandle b;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

  bool transpose_a, transpose_b;
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_a", &transpose_a));
  TF_RETURN_IF_ERROR(c->GetAttr("transpose_b", &transpose_b));
  DimensionHandle output_rows = transpose_a ? c->Dim(a, 1) : c->Dim(a, 0);
  DimensionHandle output_cols = transpose_b ? c->Dim(b, 0) : c->Dim(b, 1);

  // Validate that the inner shapes are compatible.
  DimensionHandle inner_a = transpose_a ? c->Dim(a, 0) : c->Dim(a, 1);
  DimensionHandle inner_b = transpose_b ? c->Dim(b, 1) : c->Dim(b, 0);
  DimensionHandle merged;
  TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));

  c->set_output(0, c->Matrix(output_rows, output_cols));
  return Status::OK();
}


REGISTER_OP("SGEMMOp")
	.Attr("transpose_a: bool = False")
	.Attr("transpose_b: bool = False")
    .Input("a: float32")
    .Input("b: float32")
    .Output("c: float32")
    .SetShapeFn(MatMulShape);

// Copy some stuff from here tensorflow/core/kernels/matmul_op.cc

class SGEMMOp : public OpKernel {
 public:
  explicit SGEMMOp(OpKernelConstruction* context) : OpKernel(context) {
	  OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transa_));
	  OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transb_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& a_tensor = context->input(0);
    const Tensor& b_tensor = context->input(1);
	
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(a_tensor.shape()),
			errors::InvalidArgument("SGEMM expects a 2-D matrix."));
	OP_REQUIRES(context, TensorShapeUtils::IsMatrix(b_tensor.shape()),
			errors::InvalidArgument("SGEMM expects a 2-D matrix."));

	auto ma = a_tensor.shape().dim_size(0);
	auto na = a_tensor.shape().dim_size(1);
	auto mb = b_tensor.shape().dim_size(0);
	auto nb = b_tensor.shape().dim_size(1);
	auto mc = transa_ ? na : ma;
	auto nc = transb_ ? mb : nb;
	auto kc = transa_ ? ma : na;

	OP_REQUIRES(context, transb_ ? kc == nb : kc == mb,
		errors::InvalidArgument("SGEMM inner dimensions mismatch."))

    // Create an output tensor
    Tensor* output_tensor = NULL;
	TensorShape result_shape({mc, nc});
    OP_REQUIRES_OK(context, context->allocate_output(0, result_shape, &output_tensor));

	// Create a BLAS API call
    CBLAS_ORDER Order = CblasRowMajor;
    CBLAS_TRANSPOSE TransA = transa_ ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE TransB = transb_ ? CblasTrans : CblasNoTrans;
    blasint M = mc;
    blasint N = nc;
    blasint K = kc;
    float alpha = 1.0f;
    auto A = reinterpret_cast<const float*>(a_tensor.tensor_data().data());
    blasint lda = na;
    auto B  = reinterpret_cast<const float*>(b_tensor.tensor_data().data());
    blasint ldb = nb;
    float beta = 0.0f;
    auto C = reinterpret_cast<float*>(const_cast<char*>(output_tensor->tensor_data().data()));
	blasint ldc = nc;
    cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

private:
  bool transa_;
  bool transb_;
};

REGISTER_KERNEL_BUILDER(Name("SGEMMOp").Device(DEVICE_CPU), SGEMMOp);
