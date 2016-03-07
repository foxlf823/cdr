

#ifndef SRC_Attention_H_
#define SRC_Attention_H_

#include "N3L.h"
#include "QuinLayer.h"


using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Attention {

public:
  QuinLayer<xpu> _quin_gates;
  UniLayer<xpu> _uni_gates;

public:
  Attention() {
  }

/*  virtual ~Attention() {
    // TODO Auto-generated destructor stub
  }*/

  inline void initial(int nWordDim, int nEntityDim, int nContextDim,
		  int seed = 0) {

	  _quin_gates.initial(nWordDim, nWordDim, nEntityDim, nEntityDim, nContextDim, nContextDim,
			  false, seed, 2);
	  _uni_gates.initial(nWordDim, nWordDim, false, seed + 10, 3);
  }


  inline void release() {
	  _quin_gates.release();
	  _uni_gates.release();
  }



public:


  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> >& x,
		  Tensor<xpu, 2, dtype> xFormer, Tensor<xpu, 2, dtype> xLatter,
		  Tensor<xpu, 2, dtype> xContext1, Tensor<xpu, 2, dtype> xContext2,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp, Tensor<xpu, 2, dtype> xSum,
      std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "Attention Forward error: dim invalid" << std::endl;
    }

    for (int idx = 0; idx < seq_size; idx++) {
    	_quin_gates.ComputeForwardScore(x[idx], xFormer, xLatter, xContext1, xContext2, xMExp[idx]);
    }
    _uni_gates.ComputeForwardScore(xMExp, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y += x[idx] * xPoolIndex[idx];
    }
  }


  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> >& x,
		  Tensor<xpu, 2, dtype> xFormer, Tensor<xpu, 2, dtype> xLatter,
		  Tensor<xpu, 2, dtype> xContext1, Tensor<xpu, 2, dtype> xContext2,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp,
      Tensor<xpu, 2, dtype> xSum, std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, std::vector<Tensor<xpu, 2, dtype> >& lx,
	  Tensor<xpu, 2, dtype> lxFormer, Tensor<xpu, 2, dtype> lxLatter,
	  Tensor<xpu, 2, dtype> lxContext1, Tensor<xpu, 2, dtype> lxContext2,
	  bool bclear = false) {
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);


    if(bclear){
      for (int idx = 0; idx < seq_size; idx++) {
        lx[idx] = 0.0;
        lxFormer[idx] = 0.0;
        lxLatter[idx] = 0.0;
        lxContext1[idx] = 0.0;
        lxContext2[idx] = 0.0;
      }
    }

    vector<Tensor<xpu, 2, dtype> > xMExpLoss(seq_size), xExpLoss(seq_size), xPoolIndexLoss(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      xMExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
      xExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
      xPoolIndexLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
    }

    Tensor<xpu, 2, dtype> xSumLoss = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);

    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndexLoss[idx] = ly * x[idx];
      lx[idx] += ly * xPoolIndex[idx];
    }

    for (int idx = 0; idx < seq_size; idx++) {
      xExpLoss[idx] += xPoolIndexLoss[idx] / xSum;
      xSumLoss -= xPoolIndexLoss[idx] * xExp[idx] / xSum / xSum;
    }

    sumpool_backward(xSumLoss, xExpLoss);

    _uni_gates.ComputeBackwardLoss(xMExp, xExp, xExpLoss, xMExpLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      _quin_gates.ComputeBackwardLoss(x[idx], xFormer, xLatter, xContext1, xContext2, xMExp[idx],
    		  xMExpLoss[idx],
			  lx[idx], lxFormer, lxLatter, lxContext1, lxContext2);
    }

    FreeSpace(&xSumLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(xExpLoss[idx]));
      FreeSpace(&(xPoolIndexLoss[idx]));
    }
  }


  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _quin_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _uni_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }


};

#endif /* SRC_AttentionPooling_H_ */
