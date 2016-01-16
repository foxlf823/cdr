

#ifndef SRC_EightLayer_H_
#define SRC_EightLayer_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class EightLayer {

public:

  Tensor<xpu, 2, dtype> _W1;
  Tensor<xpu, 2, dtype> _W2;
  Tensor<xpu, 2, dtype> _W3;
  Tensor<xpu, 2, dtype> _W4;
  Tensor<xpu, 2, dtype> _W5;
  Tensor<xpu, 2, dtype> _W6;
  Tensor<xpu, 2, dtype> _W7;
  Tensor<xpu, 2, dtype> _W8;
  Tensor<xpu, 2, dtype> _b;

  Tensor<xpu, 2, dtype> _gradW1;
  Tensor<xpu, 2, dtype> _gradW2;
  Tensor<xpu, 2, dtype> _gradW3;
  Tensor<xpu, 2, dtype> _gradW4;
  Tensor<xpu, 2, dtype> _gradW5;
  Tensor<xpu, 2, dtype> _gradW6;
  Tensor<xpu, 2, dtype> _gradW7;
  Tensor<xpu, 2, dtype> _gradW8;
  Tensor<xpu, 2, dtype> _gradb;

  Tensor<xpu, 2, dtype> _eg2W1;
  Tensor<xpu, 2, dtype> _eg2W2;
  Tensor<xpu, 2, dtype> _eg2W3;
  Tensor<xpu, 2, dtype> _eg2W4;
  Tensor<xpu, 2, dtype> _eg2W5;
  Tensor<xpu, 2, dtype> _eg2W6;
  Tensor<xpu, 2, dtype> _eg2W7;
  Tensor<xpu, 2, dtype> _eg2W8;
  Tensor<xpu, 2, dtype> _eg2b;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp

public:
  EightLayer() {
  }


  virtual ~EightLayer() {
    // TODO Auto-generated destructor stub
  }

  inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, int nISize4, int nISize5,
		  int nISize6, int nISize7, int nISize8,
		  bool bUseB = true, int seed = 0, int funcType = 0) {
    dtype bound = sqrt(6.0 / (nOSize + nISize1 + nISize2 + nISize3 + nISize4 + nISize5
    		+nISize6+nISize7+nISize8+ 1));

    _W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    _gradW1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    _eg2W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);

    _W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    _gradW2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    _eg2W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);

    _W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    _gradW3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    _eg2W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);

    _W4 = NewTensor<xpu>(Shape2(nOSize, nISize4), d_zero);
    _gradW4 = NewTensor<xpu>(Shape2(nOSize, nISize4), d_zero);
    _eg2W4 = NewTensor<xpu>(Shape2(nOSize, nISize4), d_zero);

    _W5 = NewTensor<xpu>(Shape2(nOSize, nISize5), d_zero);
    _gradW5 = NewTensor<xpu>(Shape2(nOSize, nISize5), d_zero);
    _eg2W5 = NewTensor<xpu>(Shape2(nOSize, nISize5), d_zero);

    _W6 = NewTensor<xpu>(Shape2(nOSize, nISize6), d_zero);
    _gradW6 = NewTensor<xpu>(Shape2(nOSize, nISize6), d_zero);
    _eg2W6 = NewTensor<xpu>(Shape2(nOSize, nISize6), d_zero);

    _W7 = NewTensor<xpu>(Shape2(nOSize, nISize7), d_zero);
    _gradW7 = NewTensor<xpu>(Shape2(nOSize, nISize7), d_zero);
    _eg2W7 = NewTensor<xpu>(Shape2(nOSize, nISize7), d_zero);

    _W8 = NewTensor<xpu>(Shape2(nOSize, nISize8), d_zero);
    _gradW8 = NewTensor<xpu>(Shape2(nOSize, nISize8), d_zero);
    _eg2W8 = NewTensor<xpu>(Shape2(nOSize, nISize8), d_zero);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    random(_W1, -1.0 * bound, 1.0 * bound, seed);
    random(_W2, -1.0 * bound, 1.0 * bound, seed+1);
    random(_W3, -1.0 * bound, 1.0 * bound, seed+2);
    random(_W4, -1.0 * bound, 1.0 * bound, seed+3);
    random(_W5, -1.0 * bound, 1.0 * bound, seed+4);
    random(_W6, -1.0 * bound, 1.0 * bound, seed+5);
    random(_W7, -1.0 * bound, 1.0 * bound, seed+6);
    random(_W8, -1.0 * bound, 1.0 * bound, seed+7);
    random(_b, -1.0 * bound, 1.0 * bound, seed+8);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> W1, Tensor<xpu, 2, dtype> W2, Tensor<xpu, 2, dtype> W3,
		  Tensor<xpu, 2, dtype> W4, Tensor<xpu, 2, dtype> W5, Tensor<xpu, 2, dtype> W6, Tensor<xpu, 2, dtype> W7,
		  Tensor<xpu, 2, dtype> W8, Tensor<xpu, 2, dtype> b, bool bUseB = true,
      int funcType = 0) {
    static int nOSize, nISize1, nISize2, nISize3,nISize4, nISize5, nISize6,nISize7, nISize8;
    nOSize = W1.size(0);
    nISize1 = W1.size(1);
    nISize2 = W2.size(1);
    nISize3 = W3.size(1);
    nISize4 = W4.size(1);
    nISize5 = W5.size(1);
    nISize6 = W6.size(1);
    nISize7 = W7.size(1);
    nISize8 = W8.size(1);

    _W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    _gradW1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    _eg2W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    Copy(_W1, W1);

    _W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    _gradW2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    _eg2W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    Copy(_W2, W2);

    _W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    _gradW3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    _eg2W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    Copy(_W3, W3);

    _W4 = NewTensor<xpu>(Shape2(nOSize, nISize4), d_zero);
    _gradW4 = NewTensor<xpu>(Shape2(nOSize, nISize4), d_zero);
    _eg2W4 = NewTensor<xpu>(Shape2(nOSize, nISize4), d_zero);
    Copy(_W4, W4);

    _W5 = NewTensor<xpu>(Shape2(nOSize, nISize5), d_zero);
    _gradW5 = NewTensor<xpu>(Shape2(nOSize, nISize5), d_zero);
    _eg2W5 = NewTensor<xpu>(Shape2(nOSize, nISize5), d_zero);
    Copy(_W5, W5);

    _W6 = NewTensor<xpu>(Shape2(nOSize, nISize6), d_zero);
    _gradW6 = NewTensor<xpu>(Shape2(nOSize, nISize6), d_zero);
    _eg2W6 = NewTensor<xpu>(Shape2(nOSize, nISize6), d_zero);
    Copy(_W6, W6);

    _W7 = NewTensor<xpu>(Shape2(nOSize, nISize7), d_zero);
    _gradW7 = NewTensor<xpu>(Shape2(nOSize, nISize7), d_zero);
    _eg2W7 = NewTensor<xpu>(Shape2(nOSize, nISize7), d_zero);
    Copy(_W7, W7);

    _W8 = NewTensor<xpu>(Shape2(nOSize, nISize8), d_zero);
    _gradW8 = NewTensor<xpu>(Shape2(nOSize, nISize8), d_zero);
    _eg2W8 = NewTensor<xpu>(Shape2(nOSize, nISize8), d_zero);
    Copy(_W8, W8);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    if (bUseB)
      Copy(_b, b);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void release() {
    FreeSpace(&_W1);
    FreeSpace(&_gradW1);
    FreeSpace(&_eg2W1);
    FreeSpace(&_W2);
    FreeSpace(&_gradW2);
    FreeSpace(&_eg2W2);
    FreeSpace(&_W3);
    FreeSpace(&_gradW3);
    FreeSpace(&_eg2W3);
    FreeSpace(&_W4);
    FreeSpace(&_gradW4);
    FreeSpace(&_eg2W4);
    FreeSpace(&_W5);
    FreeSpace(&_gradW5);
    FreeSpace(&_eg2W5);

    FreeSpace(&_W6);
    FreeSpace(&_gradW6);
    FreeSpace(&_eg2W6);

    FreeSpace(&_W7);
    FreeSpace(&_gradW7);
    FreeSpace(&_eg2W7);

    FreeSpace(&_W8);
    FreeSpace(&_gradW8);
    FreeSpace(&_eg2W8);

    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);
  }


  inline dtype squarenormAll() {
    dtype result = squarenorm(_gradW1);
    result += squarenorm(_gradW2);
    result += squarenorm(_gradW3);
    result += squarenorm(_gradW4);
    result += squarenorm(_gradW5);
    result += squarenorm(_gradW6);
    result += squarenorm(_gradW7);
    result += squarenorm(_gradW8);
    if (_bUseB) {
      result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _gradW1 = _gradW1 * scale;
    _gradW2 = _gradW2 * scale;
    _gradW3 = _gradW3 * scale;
    _gradW4 = _gradW4 * scale;
    _gradW5 = _gradW5 * scale;
    _gradW6 = _gradW6 * scale;
    _gradW7 = _gradW7 * scale;
    _gradW8 = _gradW8 * scale;
    if (_bUseB) {
      _gradb = _gradb * scale;
    }
  }

public:
  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2,
		  Tensor<xpu, 2, dtype> x3, Tensor<xpu, 2, dtype> x4, Tensor<xpu, 2, dtype> x5,
		  Tensor<xpu, 2, dtype> x6, Tensor<xpu, 2, dtype> x7, Tensor<xpu, 2, dtype> x8,
		  Tensor<xpu, 2, dtype> y) {
    y = dot(x1, _W1.T());
    y += dot(x2, _W2.T());
    y += dot(x3, _W3.T());
    y += dot(x4, _W4.T());
    y += dot(x5, _W5.T());
    y += dot(x6, _W6.T());
    y += dot(x7, _W7.T());
    y += dot(x8, _W8.T());
    if (_bUseB)
      y = y + _b;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);
    else if (_funcType == 3)
      y = F<nl_exp>(y);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2,
		  Tensor<xpu, 2, dtype> x3, Tensor<xpu, 2, dtype> x4, Tensor<xpu, 2, dtype> x5,
		  Tensor<xpu, 2, dtype> x6, Tensor<xpu, 2, dtype> x7, Tensor<xpu, 2, dtype> x8,
		  Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly,
      Tensor<xpu, 2, dtype> lx1, Tensor<xpu, 2, dtype> lx2, Tensor<xpu, 2, dtype> lx3,
	  Tensor<xpu, 2, dtype> lx4, Tensor<xpu, 2, dtype> lx5,
	  Tensor<xpu, 2, dtype> lx6, Tensor<xpu, 2, dtype> lx7,Tensor<xpu, 2, dtype> lx8,
	  bool bclear = false) {
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear) {
      lx1 = 0.0;
      lx2 = 0.0;
      lx3 = 0.0;
      lx4 = 0.0;
      lx5 = 0.0;
      lx6 = 0.0;
      lx7 = 0.0;
      lx8 = 0.0;
    }
    if (_funcType == 0) {
      deri_yx = F<nl_dtanh>(y);
      cly = ly * deri_yx;
    } else if (_funcType == 1) {
      deri_yx = F<nl_dsigmoid>(y);
      cly = ly * deri_yx;
    } else if (_funcType == 3) {
      cly = ly * y;
    } else {
      Copy(cly, ly);
    }
    //_gradW
    _gradW1 += dot(cly.T(), x1);
    _gradW2 += dot(cly.T(), x2);
    _gradW3 += dot(cly.T(), x3);
    _gradW4 += dot(cly.T(), x4);
    _gradW5 += dot(cly.T(), x5);
    _gradW6 += dot(cly.T(), x6);
    _gradW7 += dot(cly.T(), x7);
    _gradW8 += dot(cly.T(), x8);

    //_gradb
    if (_bUseB)
      _gradb += cly;

    //lx
    lx1 += dot(cly, _W1);
    lx2 += dot(cly, _W2);
    lx3 += dot(cly, _W3);
    lx4 += dot(cly, _W4);
    lx5 += dot(cly, _W5);
    lx6 += dot(cly, _W6);
    lx7 += dot(cly, _W7);
    lx8 += dot(cly, _W8);

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _gradW1 = _gradW1 + _W1 * regularizationWeight;
    _eg2W1 = _eg2W1 + _gradW1 * _gradW1;
    _W1 = _W1 - _gradW1 * adaAlpha / F<nl_sqrt>(_eg2W1 + adaEps);

    _gradW2 = _gradW2 + _W2 * regularizationWeight;
    _eg2W2 = _eg2W2 + _gradW2 * _gradW2;
    _W2 = _W2 - _gradW2 * adaAlpha / F<nl_sqrt>(_eg2W2 + adaEps);

    _gradW3 = _gradW3 + _W3 * regularizationWeight;
    _eg2W3 = _eg2W3 + _gradW3 * _gradW3;
    _W3 = _W3 - _gradW3 * adaAlpha / F<nl_sqrt>(_eg2W3 + adaEps);

    _gradW4 = _gradW4 + _W4 * regularizationWeight;
    _eg2W4 = _eg2W4 + _gradW4 * _gradW4;
    _W4 = _W4 - _gradW4 * adaAlpha / F<nl_sqrt>(_eg2W4 + adaEps);

    _gradW5 = _gradW5 + _W5 * regularizationWeight;
    _eg2W5 = _eg2W5 + _gradW5 * _gradW5;
    _W5 = _W5 - _gradW5 * adaAlpha / F<nl_sqrt>(_eg2W5 + adaEps);

    _gradW6 = _gradW6 + _W6 * regularizationWeight;
    _eg2W6 = _eg2W6 + _gradW6 * _gradW6;
    _W6 = _W6 - _gradW6 * adaAlpha / F<nl_sqrt>(_eg2W6 + adaEps);

    _gradW7 = _gradW7 + _W7 * regularizationWeight;
    _eg2W7 = _eg2W7 + _gradW7 * _gradW7;
    _W7 = _W7 - _gradW7 * adaAlpha / F<nl_sqrt>(_eg2W7 + adaEps);

    _gradW8 = _gradW8 + _W8 * regularizationWeight;
    _eg2W8 = _eg2W8 + _gradW8 * _gradW8;
    _W8 = _W8 - _gradW8 * adaAlpha / F<nl_sqrt>(_eg2W8 + adaEps);

    if (_bUseB) {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }

    clearGrad();
  }

  inline void clearGrad() {
    _gradW1 = 0;
    _gradW2 = 0;
    _gradW3 = 0;
    _gradW4 = 0;
    _gradW5 = 0;
    _gradW6 = 0;
    _gradW7 = 0;
    _gradW8 = 0;
    if (_bUseB)
      _gradb = 0;
  }


  
};

#endif
