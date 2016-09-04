
#ifndef SRC_PoolGRNNClassifier4_H_
#define SRC_PoolGRNNClassifier4_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Feature.h"
#include "N3L.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class PoolGRNNClassifier4 {
public:
	PoolGRNNClassifier4() {
    _dropOut = 0.5;
  }
  ~PoolGRNNClassifier4() {

  }

public:
  LookupTable<xpu> _words;
  LookupTable<xpu> _pos;
  LookupTable<xpu> _sst;
  LookupTable<xpu> _ner;

  int _wordcontext, _wordwindow;
  int _wordSize;
  int _wordDim;

  int _token_representation_size;
  int _inputsize;
  int _hiddensize;
  int _rnnHiddenSize;


  UniLayer<xpu> _olayer_linear;
  UniLayer<xpu> _tanh_project;

  GRNN<xpu> _rnn_left;
  GRNN<xpu> _rnn_right;

  GRNN<xpu> _pos_left;
  GRNN<xpu> _pos_right;
  UniLayer<xpu> _pos_project;
  GRNN<xpu> _sst_left;
  GRNN<xpu> _sst_right;
  UniLayer<xpu> _sst_project;
  GRNN<xpu> _ner_left;
  GRNN<xpu> _ner_right;
  UniLayer<xpu> _ner_project;


  int _poolmanners;
  int _poolfunctions;


  int _poolsize;


  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3, min, 4, std, 5, pro

  Options options;

  int _poolInputSize;

  int _otherInputSize;
  int _channel;
  int _otherDim;

public:

  inline void init(const NRMat<dtype>& wordEmb, Options options) {
	  this->options = options;
    _wordcontext = options.wordcontext;
    _wordwindow = 2 * _wordcontext + 1;
    _wordSize = wordEmb.nrows();
    _wordDim = wordEmb.ncols();

    _labelSize = MAX_RELATION;
    _token_representation_size = _wordDim;
    _poolfunctions = 5;
    _poolmanners = _poolfunctions * 5; //( left, right, target) * (avg, max, min, std, pro)
    _inputsize = _wordwindow * _token_representation_size;
    _hiddensize = options.wordEmbSize;
    _rnnHiddenSize = options.rnnHiddenSize;

    _channel = options.channelMode;
    _otherDim = options.otherEmbSize;
    _otherInputSize = 0;
	if((_channel & 2) == 2) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 4) == 4) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 8) == 8) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 16) == 16) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 32) == 32) {
		_otherInputSize += _otherDim;
	}

    _poolInputSize = _hiddensize + _otherInputSize;
    _poolsize = _poolmanners * _poolInputSize;



    _words.initial(wordEmb);


    _rnn_left.initial(_rnnHiddenSize, _inputsize, true, 10);
    _rnn_right.initial(_rnnHiddenSize, _inputsize, false, 40);

	if((_channel & 2) == 2) {
	}
	if((_channel & 4) == 4) {
	}
	if((_channel & 8) == 8) {
	    _ner_left.initial(_otherDim, _otherDim, true, 210);
	    _ner_right.initial(_otherDim, _otherDim, false, 220);
	    _ner_project.initial(_otherDim, 2*_otherDim, true, 230, 0);
	}
	if((_channel & 16) == 16) {
	    _pos_left.initial(_otherDim, _otherDim, true, 240);
	    _pos_right.initial(_otherDim, _otherDim, false, 250);
	    _pos_project.initial(_otherDim, 2*_otherDim, true, 260, 0);
	}
	if((_channel & 32) == 32) {
	    _sst_left.initial(_otherDim, _otherDim, true, 270);
	    _sst_right.initial(_otherDim, _otherDim, false, 280);
	    _sst_project.initial(_otherDim, 2*_otherDim, true, 290, 0);
	}


    _tanh_project.initial(_hiddensize, 2*_rnnHiddenSize, true, 70, 0);
    _olayer_linear.initial(_labelSize, _poolsize , false, 80, 2);

    _remove = 0;

    cout<<"PoolGRNNClassifier4 initial"<<endl;
    cout<< "External features use their own GRNN whose outputs are appended in the project layer"<<endl;
  }

  inline void release() {
    _words.release();
    _sst.release();
    _ner.release();
    _pos.release();

    _olayer_linear.release();
    _tanh_project.release();
    _rnn_left.release();
    _rnn_right.release();

	if((_channel & 2) == 2) {
	}
	if((_channel & 4) == 4) {
	}
	if((_channel & 8) == 8) {
	    _ner_left.release();
	    _ner_right.release();
	    _ner_project.release();
	}
	if((_channel & 16) == 16) {
	    _pos_left.release();
	    _pos_right.release();
	    _pos_project.release();
	}
	if((_channel & 32) == 32) {
	    _sst_left.release();
	    _sst_right.release();
	    _sst_project.release();
	}


  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();

      Tensor<xpu, 3, dtype> input, inputLoss;
      Tensor<xpu, 3, dtype> project, projectLoss;

      Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current;
      Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current;

      Tensor<xpu, 3, dtype> rnn_hidden_merge, rnn_hidden_mergeLoss;


      Tensor<xpu, 3, dtype> posprime, posprimeLoss, posprimeMask;
      Tensor<xpu, 3, dtype> sstprime, sstprimeLoss, sstprimeMask;
      Tensor<xpu, 3, dtype> nerprime, nerprimeLoss, nerprimeMask;

      Tensor<xpu, 3, dtype> pos_hidden_left, pos_hidden_leftLoss;
      Tensor<xpu, 3, dtype> pos_hidden_left_reset, pos_hidden_left_afterreset, pos_hidden_left_update, pos_hidden_left_current;
      Tensor<xpu, 3, dtype> pos_hidden_right, pos_hidden_rightLoss;
      Tensor<xpu, 3, dtype> pos_hidden_right_reset, pos_hidden_right_afterreset, pos_hidden_right_update, pos_hidden_right_current;
      Tensor<xpu, 3, dtype> pos_hidden_merge, pos_hidden_mergeLoss;
      Tensor<xpu, 3, dtype> pos_project, pos_projectLoss;

      Tensor<xpu, 3, dtype> sst_hidden_left, sst_hidden_leftLoss;
      Tensor<xpu, 3, dtype> sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current;
      Tensor<xpu, 3, dtype> sst_hidden_right, sst_hidden_rightLoss;
      Tensor<xpu, 3, dtype> sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current;
      Tensor<xpu, 3, dtype> sst_hidden_merge, sst_hidden_mergeLoss;
      Tensor<xpu, 3, dtype> sst_project, sst_projectLoss;

      Tensor<xpu, 3, dtype> ner_hidden_left, ner_hidden_leftLoss;
      Tensor<xpu, 3, dtype> ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current;
      Tensor<xpu, 3, dtype> ner_hidden_right, ner_hidden_rightLoss;
      Tensor<xpu, 3, dtype> ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current;
      Tensor<xpu, 3, dtype> ner_hidden_merge, ner_hidden_mergeLoss;
      Tensor<xpu, 3, dtype> ner_project, ner_projectLoss;

      Tensor<xpu, 3, dtype> poolInput, poolInputLoss;

      vector<Tensor<xpu, 2, dtype> > pool(_poolmanners), poolLoss(_poolmanners);
      vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

      Tensor<xpu, 2, dtype> poolmerge, poolmergeLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;


      Tensor<xpu, 3, dtype> wordprime, wordprimeLoss, wordprimeMask;
      Tensor<xpu, 3, dtype> wordrepresent, wordrepresentLoss;

      hash_set<int> beforeIndex, formerIndex, middleIndex, latterIndex, afterIndex;
      Tensor<xpu, 2, dtype> beforerepresent, beforerepresentLoss;
      Tensor<xpu, 2, dtype> formerrepresent, formerrepresentLoss;
	  Tensor<xpu, 2, dtype> middlerepresent, middlerepresentLoss;
      Tensor<xpu, 2, dtype> latterrepresent, latterrepresentLoss;
	  Tensor<xpu, 2, dtype> afterrepresent, afterrepresentLoss;

      //initialize
      wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
      wordprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 1.0);
      wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);
      wordrepresentLoss = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

      input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);
      inputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

      rnn_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

      rnn_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
      rnn_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

      rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_rnnHiddenSize), 0.0);
      rnn_hidden_mergeLoss = NewTensor<xpu>(Shape3(seq_size, 1, 2*_rnnHiddenSize), 0.0);

      project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);
      projectLoss = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);


  	if((_channel & 2) == 2) {
  	}
  	if((_channel & 4) == 4) {
  	}
  	if((_channel & 8) == 8) {
        nerprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        nerprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        nerprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);
        ner_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        ner_hidden_mergeLoss = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        ner_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_projectLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
  	}
  	if((_channel & 16) == 16) {
        posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        posprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        posprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);
        pos_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        pos_hidden_mergeLoss = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        pos_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_projectLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
  	}
  	if((_channel & 32) == 32) {
        sstprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sstprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sstprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);
        sst_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_leftLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_rightLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        sst_hidden_mergeLoss = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        sst_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_projectLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
  	}

      poolInput = NewTensor<xpu>(Shape3(seq_size, 1, _poolInputSize), 0.0);
      poolInputLoss = NewTensor<xpu>(Shape3(seq_size, 1, _poolInputSize), 0.0);

      for (int idm = 0; idm < _poolmanners; idm++) {
        pool[idm] = NewTensor<xpu>(Shape2(1, _poolInputSize), 0.0);
        poolLoss[idm] = NewTensor<xpu>(Shape2(1, _poolInputSize), 0.0);
        poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _poolInputSize), 0.0);
      }

      beforerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
      beforerepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);

      formerrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
      formerrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);

      middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
      middlerepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);

	  latterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
      latterrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);

	  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
      afterrepresentLoss = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);


      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      outputLoss = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);
      //forward propagation
      //input setting, and linear setting
      for (int idx = 0; idx < seq_size; idx++) {
        const Feature& feature = example.m_features[idx];
        //linear features should not be dropped out

        srand(iter * example_num + count * seq_size + idx);

        const vector<int>& words = feature.words;
        if (idx < example.formerTkBegin) {
          beforeIndex.insert(idx);
        } else if(idx >= example.formerTkBegin && idx <= example.formerTkEnd) {
			formerIndex.insert(idx);
		} else if (idx >= example.latterTkBegin && idx <= example.latterTkEnd) {
          latterIndex.insert(idx);
        } else if (idx > example.latterTkEnd) {
          afterIndex.insert(idx);
        } else {
          middleIndex.insert(idx);
        }

       _words.GetEmb(words[0], wordprime[idx]);
        //dropout
        dropoutcol(wordprimeMask[idx], _dropOut);
        wordprime[idx] = wordprime[idx] * wordprimeMask[idx];

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
		   _ner.GetEmb(feature.ner, nerprime[idx]);
			//dropout
			dropoutcol(nerprimeMask[idx], _dropOut);
			nerprime[idx] = nerprime[idx] * nerprimeMask[idx];
		}
		if((_channel & 16) == 16) {
		   _pos.GetEmb(feature.pos, posprime[idx]);
			//dropout
			dropoutcol(posprimeMask[idx], _dropOut);
			posprime[idx] = posprime[idx] * posprimeMask[idx];
		}
		if((_channel & 32) == 32) {
		   _sst.GetEmb(feature.sst, sstprime[idx]);
			//dropout
			dropoutcol(sstprimeMask[idx], _dropOut);
			sstprime[idx] = sstprime[idx] * sstprimeMask[idx];
		}
      }

      for (int idx = 0; idx < seq_size; idx++) {
        wordrepresent[idx] += wordprime[idx];
      }

      windowlized(wordrepresent, input, _wordcontext);

      _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
      _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

		if((_channel & 2) == 2) {
		}
		if((_channel & 4) == 4) {
		}
		if((_channel & 8) == 8) {
		      _ner_left.ComputeForwardScore(nerprime, ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current, ner_hidden_left);
		      _ner_right.ComputeForwardScore(nerprime, ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current, ner_hidden_right);
		}
		if((_channel & 16) == 16) {
		      _pos_left.ComputeForwardScore(posprime, pos_hidden_left_reset, pos_hidden_left_afterreset, pos_hidden_left_update, pos_hidden_left_current, pos_hidden_left);
		      _pos_right.ComputeForwardScore(posprime, pos_hidden_right_reset, pos_hidden_right_afterreset, pos_hidden_right_update, pos_hidden_right_current, pos_hidden_right);
		}
		if((_channel & 32) == 32) {
		      _sst_left.ComputeForwardScore(sstprime, sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current, sst_hidden_left);
		      _sst_right.ComputeForwardScore(sstprime, sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current, sst_hidden_right);
		}

      for (int idx = 0; idx < seq_size; idx++) {
        concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);

		if((_channel & 2) == 2) {
		}
		if((_channel & 4) == 4) {
		}
		if((_channel & 8) == 8) {
	        concat(ner_hidden_left[idx], ner_hidden_right[idx], ner_hidden_merge[idx]);
		}
		if((_channel & 16) == 16) {
	        concat(pos_hidden_left[idx], pos_hidden_right[idx], pos_hidden_merge[idx]);
		}
		if((_channel & 32) == 32) {
	        concat(sst_hidden_left[idx], sst_hidden_right[idx], sst_hidden_merge[idx]);
		}
      }

      // do we need a convolution? future work, currently needn't
      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);

		if((_channel & 2) == 2) {
		}
		if((_channel & 4) == 4) {
		}
		if((_channel & 8) == 8) {
			_ner_project.ComputeForwardScore(ner_hidden_merge[idx], ner_project[idx]);
		}
		if((_channel & 16) == 16) {
			_pos_project.ComputeForwardScore(pos_hidden_merge[idx], pos_project[idx]);
		}
		if((_channel & 32) == 32) {
			_sst_project.ComputeForwardScore(sst_hidden_merge[idx], sst_project[idx]);
		}
      }


      for(int i=0;i<seq_size;i++) {
          vector<Tensor<xpu, 2, dtype> > v_otherInput;
          v_otherInput.push_back(project[i]);

    		if((_channel & 2) == 2) {

    		}
    		if((_channel & 4) == 4) {

    		}
    		if((_channel & 8) == 8) {
    			  v_otherInput.push_back(ner_project[i]);
    		}
    		if((_channel & 16) == 16) {
    			  v_otherInput.push_back(pos_project[i]);
    		}
    		if((_channel & 32) == 32) {
    			  v_otherInput.push_back(sst_project[i]);
    		}

    		concat(v_otherInput, poolInput[i]);
      }

      offset = 0;
      //before
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(poolInput, pool[offset], poolIndex[offset], beforeIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], beforeIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], beforeIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], beforeIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], beforeIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], beforerepresent);

      offset = _poolfunctions;
      //former
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(poolInput, pool[offset], poolIndex[offset], formerIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], formerIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], formerIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], formerIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], formerIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], formerrepresent);

      offset = 2 * _poolfunctions;
      //middle
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(poolInput, pool[offset], poolIndex[offset], middleIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], middleIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], middleIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], middleIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], middleIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], middlerepresent);

	  offset = 3 * _poolfunctions;
      //latter
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(poolInput, pool[offset], poolIndex[offset], latterIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], latterIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], latterIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], latterIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], latterIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], latterrepresent);

	  offset = 4 * _poolfunctions;
      //after
      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        avgpool_forward(poolInput, pool[offset], poolIndex[offset], afterIndex);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], afterIndex);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], afterIndex);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], afterIndex);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], afterIndex);
      }

      concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], afterrepresent);


      concat(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent, poolmerge);


      _olayer_linear.ComputeForwardScore(poolmerge, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _olayer_linear.ComputeBackwardLoss(poolmerge, output, outputLoss, poolmergeLoss);

      unconcat(beforerepresentLoss, formerrepresentLoss, middlerepresentLoss, latterrepresentLoss, afterrepresentLoss, poolmergeLoss);


      offset = 0;
      //before
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], beforerepresentLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  poolInputLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], poolInputLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], poolInputLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], poolInputLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], poolInputLoss);
      }

      offset = _poolfunctions;
      //former
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], formerrepresentLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  poolInputLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], poolInputLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], poolInputLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], poolInputLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], poolInputLoss);
      }

      offset = 2 * _poolfunctions;
      //middle
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], middlerepresentLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  poolInputLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], poolInputLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], poolInputLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], poolInputLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], poolInputLoss);
      }

	  offset = 3 * _poolfunctions;
      //latter
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], latterrepresentLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  poolInputLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], poolInputLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], poolInputLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], poolInputLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], poolInputLoss);
      }

	  offset = 4 * _poolfunctions;
      //after
      unconcat(poolLoss[offset], poolLoss[offset + 1], poolLoss[offset + 2], poolLoss[offset + 3], poolLoss[offset + 4], afterrepresentLoss);

      //avg pooling
      if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
        pool_backward(poolLoss[offset], poolIndex[offset],  poolInputLoss);
      }
      //max pooling
      if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
        pool_backward(poolLoss[offset + 1], poolIndex[offset + 1], poolInputLoss);
      }
      //min pooling
      if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
        pool_backward(poolLoss[offset + 2], poolIndex[offset + 2], poolInputLoss);
      }
      //std pooling
      if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
        pool_backward(poolLoss[offset + 3], poolIndex[offset + 3], poolInputLoss);
      }
      //pro pooling
      if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
        pool_backward(poolLoss[offset + 4], poolIndex[offset + 4], poolInputLoss);
      }

      for(int i=0;i<seq_size;i++) {
    	  vector<Tensor<xpu, 2, dtype> > v_otherInputLoss;
    	  v_otherInputLoss.push_back(projectLoss[i]);

    		if((_channel & 2) == 2) {

    		}
    		if((_channel & 4) == 4) {

    		}
    		if((_channel & 8) == 8) {
    			v_otherInputLoss.push_back(ner_projectLoss[i]);
    		}
    		if((_channel & 16) == 16) {
    			v_otherInputLoss.push_back(pos_projectLoss[i]);
    		}
    		if((_channel & 32) == 32) {
    			v_otherInputLoss.push_back(sst_projectLoss[i]);
    		}

    		unconcat(v_otherInputLoss, poolInputLoss[i]);
      }


      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeBackwardLoss(rnn_hidden_merge[idx], project[idx], projectLoss[idx], rnn_hidden_mergeLoss[idx]);

    	if((_channel & 2) == 2) {
    	}
    	if((_channel & 4) == 4) {
    	}
    	if((_channel & 8) == 8) {
            _ner_project.ComputeBackwardLoss(ner_hidden_merge[idx], ner_project[idx], ner_projectLoss[idx], ner_hidden_mergeLoss[idx]);
    	}
    	if((_channel & 16) == 16) {
            _pos_project.ComputeBackwardLoss(pos_hidden_merge[idx], pos_project[idx], pos_projectLoss[idx], pos_hidden_mergeLoss[idx]);
    	}
    	if((_channel & 32) == 32) {
            _sst_project.ComputeBackwardLoss(sst_hidden_merge[idx], sst_project[idx], sst_projectLoss[idx], sst_hidden_mergeLoss[idx]);
    	}
      }

      for (int idx = 0; idx < seq_size; idx++) {
        unconcat(rnn_hidden_leftLoss[idx], rnn_hidden_rightLoss[idx], rnn_hidden_mergeLoss[idx]);

    	if((_channel & 2) == 2) {
    	}
    	if((_channel & 4) == 4) {
    	}
    	if((_channel & 8) == 8) {
    		unconcat(ner_hidden_leftLoss[idx], ner_hidden_rightLoss[idx], ner_hidden_mergeLoss[idx]);
    	}
    	if((_channel & 16) == 16) {
    		unconcat(pos_hidden_leftLoss[idx], pos_hidden_rightLoss[idx], pos_hidden_mergeLoss[idx]);
    	}
    	if((_channel & 32) == 32) {
    		unconcat(sst_hidden_leftLoss[idx], sst_hidden_rightLoss[idx], sst_hidden_mergeLoss[idx]);
    	}
      }

      _rnn_left.ComputeBackwardLoss(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left, rnn_hidden_leftLoss, inputLoss);
      _rnn_right.ComputeBackwardLoss(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right, rnn_hidden_rightLoss, inputLoss);

  	if((_channel & 2) == 2) {
  	}
  	if((_channel & 4) == 4) {
  	}
  	if((_channel & 8) == 8) {
        _ner_left.ComputeBackwardLoss(nerprime, ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current, ner_hidden_left, ner_hidden_leftLoss, nerprimeLoss);
        _ner_right.ComputeBackwardLoss(nerprime, ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current, ner_hidden_right, ner_hidden_rightLoss, nerprimeLoss);
  	}
  	if((_channel & 16) == 16) {
        _pos_left.ComputeBackwardLoss(posprime, pos_hidden_left_reset, pos_hidden_left_afterreset, pos_hidden_left_update, pos_hidden_left_current, pos_hidden_left, pos_hidden_leftLoss, posprimeLoss);
        _pos_right.ComputeBackwardLoss(posprime, pos_hidden_right_reset, pos_hidden_right_afterreset, pos_hidden_right_update, pos_hidden_right_current, pos_hidden_right, pos_hidden_rightLoss, posprimeLoss);
  	}
  	if((_channel & 32) == 32) {
        _sst_left.ComputeBackwardLoss(sstprime, sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current, sst_hidden_left, sst_hidden_leftLoss, sstprimeLoss);
        _sst_right.ComputeBackwardLoss(sstprime, sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current, sst_hidden_right, sst_hidden_rightLoss, sstprimeLoss);
  	}
	  
      // word context
      windowlized_backward(wordrepresentLoss, inputLoss, _wordcontext);

      for (int idx = 0; idx < seq_size; idx++) {
        wordprimeLoss[idx] += wordrepresentLoss[idx];
      }

      if (_words.bEmbFineTune()) {
        for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];
          const vector<int>& words = feature.words;
          wordprimeLoss[idx] = wordprimeLoss[idx] * wordprimeMask[idx];
          _words.EmbLoss(words[0], wordprimeLoss[idx]);
        }
      }

      for (int idx = 0; idx < seq_size; idx++) {
          const Feature& feature = example.m_features[idx];

  		if((_channel & 2) == 2) {

  		}
  		if((_channel & 4) == 4) {

  		}
  		if((_channel & 8) == 8) {
  	          nerprimeLoss[idx] = nerprimeLoss[idx] * nerprimeMask[idx];
  			_ner.EmbLoss(feature.ner, nerprimeLoss[idx]);
  		}
  		if((_channel & 16) == 16) {
	          posprimeLoss[idx] = posprimeLoss[idx] * posprimeMask[idx];
			_pos.EmbLoss(feature.pos, posprimeLoss[idx]);
  		}
  		if((_channel & 32) == 32) {
	          sstprimeLoss[idx] = sstprimeLoss[idx] * sstprimeMask[idx];
			_sst.EmbLoss(feature.sst, sstprimeLoss[idx]);
  		}
      }



      //release
      FreeSpace(&wordprime);
      FreeSpace(&wordprimeLoss);
      FreeSpace(&wordprimeMask);
      FreeSpace(&wordrepresent);
      FreeSpace(&wordrepresentLoss);

      FreeSpace(&input);
      FreeSpace(&inputLoss);

      FreeSpace(&rnn_hidden_left_reset);
      FreeSpace(&rnn_hidden_left_update);
      FreeSpace(&rnn_hidden_left_afterreset);
      FreeSpace(&rnn_hidden_left_current);
      FreeSpace(&rnn_hidden_left);
      FreeSpace(&rnn_hidden_leftLoss);

      FreeSpace(&rnn_hidden_right_reset);
      FreeSpace(&rnn_hidden_right_update);
      FreeSpace(&rnn_hidden_right_afterreset);
      FreeSpace(&rnn_hidden_right_current);
      FreeSpace(&rnn_hidden_right);
      FreeSpace(&rnn_hidden_rightLoss);

      FreeSpace(&rnn_hidden_merge);
      FreeSpace(&rnn_hidden_mergeLoss);

  	if((_channel & 2) == 2) {
  	}
  	if((_channel & 4) == 4) {
  	}
  	if((_channel & 8) == 8) {
        FreeSpace(&nerprime);
        FreeSpace(&nerprimeLoss);
        FreeSpace(&nerprimeMask);
        FreeSpace(&ner_hidden_left_reset);
        FreeSpace(&ner_hidden_left_update);
        FreeSpace(&ner_hidden_left_afterreset);
        FreeSpace(&ner_hidden_left_current);
        FreeSpace(&ner_hidden_left);
        FreeSpace(&ner_hidden_leftLoss);
        FreeSpace(&ner_hidden_right_reset);
        FreeSpace(&ner_hidden_right_update);
        FreeSpace(&ner_hidden_right_afterreset);
        FreeSpace(&ner_hidden_right_current);
        FreeSpace(&ner_hidden_right);
        FreeSpace(&ner_hidden_rightLoss);
        FreeSpace(&ner_hidden_merge);
        FreeSpace(&ner_hidden_mergeLoss);
        FreeSpace(&ner_project);
        FreeSpace(&ner_projectLoss);
  	}
  	if((_channel & 16) == 16) {
        FreeSpace(&posprime);
        FreeSpace(&posprimeLoss);
        FreeSpace(&posprimeMask);
  	     FreeSpace(&pos_hidden_left_reset);
  	      FreeSpace(&pos_hidden_left_update);
  	      FreeSpace(&pos_hidden_left_afterreset);
  	      FreeSpace(&pos_hidden_left_current);
  	      FreeSpace(&pos_hidden_left);
  	      FreeSpace(&pos_hidden_leftLoss);
  	      FreeSpace(&pos_hidden_right_reset);
  	      FreeSpace(&pos_hidden_right_update);
  	      FreeSpace(&pos_hidden_right_afterreset);
  	      FreeSpace(&pos_hidden_right_current);
  	      FreeSpace(&pos_hidden_right);
  	      FreeSpace(&pos_hidden_rightLoss);
  	      FreeSpace(&pos_hidden_merge);
  	      FreeSpace(&pos_hidden_mergeLoss);
  	      FreeSpace(&pos_project);
  	      FreeSpace(&pos_projectLoss);
  	}
  	if((_channel & 32) == 32) {
        FreeSpace(&sstprime);
        FreeSpace(&sstprimeLoss);
        FreeSpace(&sstprimeMask);
        FreeSpace(&sst_hidden_left_reset);
        FreeSpace(&sst_hidden_left_update);
        FreeSpace(&sst_hidden_left_afterreset);
        FreeSpace(&sst_hidden_left_current);
        FreeSpace(&sst_hidden_left);
        FreeSpace(&sst_hidden_leftLoss);
        FreeSpace(&sst_hidden_right_reset);
        FreeSpace(&sst_hidden_right_update);
        FreeSpace(&sst_hidden_right_afterreset);
        FreeSpace(&sst_hidden_right_current);
        FreeSpace(&sst_hidden_right);
        FreeSpace(&sst_hidden_rightLoss);
        FreeSpace(&sst_hidden_merge);
        FreeSpace(&sst_hidden_mergeLoss);
        FreeSpace(&sst_project);
        FreeSpace(&sst_projectLoss);
  	}


      FreeSpace(&poolInput);
      FreeSpace(&poolInputLoss);

      FreeSpace(&project);
      FreeSpace(&projectLoss);

      for (int idm = 0; idm < _poolmanners; idm++) {
        FreeSpace(&(pool[idm]));
        FreeSpace(&(poolLoss[idm]));
        FreeSpace(&(poolIndex[idm]));
      }


      FreeSpace(&beforerepresent);
      FreeSpace(&beforerepresentLoss);
      FreeSpace(&formerrepresent);
      FreeSpace(&formerrepresentLoss);
      FreeSpace(&middlerepresent);
      FreeSpace(&middlerepresentLoss);
	  FreeSpace(&latterrepresent);
      FreeSpace(&latterrepresentLoss);
	  FreeSpace(&afterrepresent);
      FreeSpace(&afterrepresentLoss);

      FreeSpace(&poolmerge);
      FreeSpace(&poolmergeLoss);
      FreeSpace(&output);
      FreeSpace(&outputLoss);

    }

    if (_eval.getAccuracy() < 0) {
      std::cout << "strange" << std::endl;
    }

    return cost;
  }

  int predict(const Example& example, vector<dtype>& results) {
		const vector<Feature>& features = example.m_features;
    int seq_size = features.size();
    int offset = 0;

    Tensor<xpu, 3, dtype> input;
    Tensor<xpu, 3, dtype> project;

    Tensor<xpu, 3, dtype> rnn_hidden_left_update;
    Tensor<xpu, 3, dtype> rnn_hidden_left_reset;
    Tensor<xpu, 3, dtype> rnn_hidden_left;
    Tensor<xpu, 3, dtype> rnn_hidden_left_afterreset;
    Tensor<xpu, 3, dtype> rnn_hidden_left_current;

    Tensor<xpu, 3, dtype> rnn_hidden_right_update;
    Tensor<xpu, 3, dtype> rnn_hidden_right_reset;
    Tensor<xpu, 3, dtype> rnn_hidden_right;
    Tensor<xpu, 3, dtype> rnn_hidden_right_afterreset;
    Tensor<xpu, 3, dtype> rnn_hidden_right_current;

    Tensor<xpu, 3, dtype> rnn_hidden_merge;

    Tensor<xpu, 3, dtype> posprime;
    Tensor<xpu, 3, dtype> sstprime;
    Tensor<xpu, 3, dtype> nerprime;

    Tensor<xpu, 3, dtype> pos_hidden_left;
    Tensor<xpu, 3, dtype> pos_hidden_left_reset, pos_hidden_left_afterreset, pos_hidden_left_update, pos_hidden_left_current;
    Tensor<xpu, 3, dtype> pos_hidden_right;
    Tensor<xpu, 3, dtype> pos_hidden_right_reset, pos_hidden_right_afterreset, pos_hidden_right_update, pos_hidden_right_current;
    Tensor<xpu, 3, dtype> pos_hidden_merge;
    Tensor<xpu, 3, dtype> pos_project;

    Tensor<xpu, 3, dtype> sst_hidden_left;
    Tensor<xpu, 3, dtype> sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current;
    Tensor<xpu, 3, dtype> sst_hidden_right;
    Tensor<xpu, 3, dtype> sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current;
    Tensor<xpu, 3, dtype> sst_hidden_merge;
    Tensor<xpu, 3, dtype> sst_project;

    Tensor<xpu, 3, dtype> ner_hidden_left;
    Tensor<xpu, 3, dtype> ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current;
    Tensor<xpu, 3, dtype> ner_hidden_right;
    Tensor<xpu, 3, dtype> ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current;
    Tensor<xpu, 3, dtype> ner_hidden_merge;
    Tensor<xpu, 3, dtype> ner_project;

    Tensor<xpu, 3, dtype> poolInput;

    vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
    vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

    Tensor<xpu, 2, dtype> poolmerge;
    Tensor<xpu, 2, dtype> output;


    Tensor<xpu, 3, dtype> wordprime, wordrepresent;

	hash_set<int> beforeIndex, formerIndex, middleIndex, latterIndex, afterIndex;
      Tensor<xpu, 2, dtype> beforerepresent;
      Tensor<xpu, 2, dtype> formerrepresent;
	  Tensor<xpu, 2, dtype> middlerepresent;
      Tensor<xpu, 2, dtype> latterrepresent;
	  Tensor<xpu, 2, dtype> afterrepresent;


    //initialize
    wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
    wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

    input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

    rnn_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
    rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

    rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_rnnHiddenSize), 0.0);

    project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);


  	if((_channel & 2) == 2) {
  	}
  	if((_channel & 4) == 4) {
  	}
  	if((_channel & 8) == 8) {
        nerprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        ner_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        ner_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
  	}
  	if((_channel & 16) == 16) {
        posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        pos_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        pos_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
  	}
  	if((_channel & 32) == 32) {
        sstprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        sst_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
        sst_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
  	}

    poolInput = NewTensor<xpu>(Shape3(seq_size, 1, _poolInputSize), 0.0);

    for (int idm = 0; idm < _poolmanners; idm++) {
      pool[idm] = NewTensor<xpu>(Shape2(1, _poolInputSize), 0.0);
      poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _poolInputSize), 0.0);
    }

    beforerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
    formerrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
    middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
	  latterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
	  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);



    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

    //forward propagation
    //input setting, and linear setting
    for (int idx = 0; idx < seq_size; idx++) {
      const Feature& feature = features[idx];
      //linear features should not be dropped out

      const vector<int>& words = feature.words;
	    if (idx < example.formerTkBegin) {
        beforeIndex.insert(idx);
      } else if(idx >= example.formerTkBegin && idx <= example.formerTkEnd) {
			formerIndex.insert(idx);
		} else if (idx >= example.latterTkBegin && idx <= example.latterTkEnd) {
        latterIndex.insert(idx);
      } else if (idx > example.latterTkEnd) {
        afterIndex.insert(idx);
      } else {
        middleIndex.insert(idx);
      }

      _words.GetEmb(words[0], wordprime[idx]);

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
		   _ner.GetEmb(feature.ner, nerprime[idx]);
		}
		if((_channel & 16) == 16) {
		   _pos.GetEmb(feature.pos, posprime[idx]);
		}
		if((_channel & 32) == 32) {
		   _sst.GetEmb(feature.sst, sstprime[idx]);
		}

    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, input, _wordcontext);

    _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
    _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

	if((_channel & 2) == 2) {
	}
	if((_channel & 4) == 4) {
	}
	if((_channel & 8) == 8) {
	      _ner_left.ComputeForwardScore(nerprime, ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current, ner_hidden_left);
	      _ner_right.ComputeForwardScore(nerprime, ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current, ner_hidden_right);
	}
	if((_channel & 16) == 16) {
	      _pos_left.ComputeForwardScore(posprime, pos_hidden_left_reset, pos_hidden_left_afterreset, pos_hidden_left_update, pos_hidden_left_current, pos_hidden_left);
	      _pos_right.ComputeForwardScore(posprime, pos_hidden_right_reset, pos_hidden_right_afterreset, pos_hidden_right_update, pos_hidden_right_current, pos_hidden_right);
	}
	if((_channel & 32) == 32) {
	      _sst_left.ComputeForwardScore(sstprime, sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current, sst_hidden_left);
	      _sst_right.ComputeForwardScore(sstprime, sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current, sst_hidden_right);
	}

    for (int idx = 0; idx < seq_size; idx++) {
        concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);

		if((_channel & 2) == 2) {
		}
		if((_channel & 4) == 4) {
		}
		if((_channel & 8) == 8) {
	        concat(ner_hidden_left[idx], ner_hidden_right[idx], ner_hidden_merge[idx]);
		}
		if((_channel & 16) == 16) {
	        concat(pos_hidden_left[idx], pos_hidden_right[idx], pos_hidden_merge[idx]);
		}
		if((_channel & 32) == 32) {
	        concat(sst_hidden_left[idx], sst_hidden_right[idx], sst_hidden_merge[idx]);
		}
    }

    // do we need a convolution? future work, currently needn't
    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);

		if((_channel & 2) == 2) {
		}
		if((_channel & 4) == 4) {
		}
		if((_channel & 8) == 8) {
			_ner_project.ComputeForwardScore(ner_hidden_merge[idx], ner_project[idx]);
		}
		if((_channel & 16) == 16) {
			_pos_project.ComputeForwardScore(pos_hidden_merge[idx], pos_project[idx]);
		}
		if((_channel & 32) == 32) {
			_sst_project.ComputeForwardScore(sst_hidden_merge[idx], sst_project[idx]);
		}
    }

    for(int i=0;i<seq_size;i++) {
        vector<Tensor<xpu, 2, dtype> > v_otherInput;
        v_otherInput.push_back(project[i]);

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
			  v_otherInput.push_back(ner_project[i]);
		}
		if((_channel & 16) == 16) {
			  v_otherInput.push_back(pos_project[i]);
		}
		if((_channel & 32) == 32) {
			  v_otherInput.push_back(sst_project[i]);
		}

  		concat(v_otherInput, poolInput[i]);
    }


    offset = 0;
    //before
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(poolInput, pool[offset], poolIndex[offset], beforeIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], beforeIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], beforeIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], beforeIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], beforeIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], beforerepresent);

    offset = _poolfunctions;
    //former
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(poolInput, pool[offset], poolIndex[offset], formerIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], formerIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], formerIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], formerIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], formerIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], formerrepresent);

    offset = 2 * _poolfunctions;
    //middle
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(poolInput, pool[offset], poolIndex[offset], middleIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], middleIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], middleIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], middleIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], middleIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], middlerepresent);

	  offset = 3 * _poolfunctions;
    //latter
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(poolInput, pool[offset], poolIndex[offset], latterIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], latterIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], latterIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], latterIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], latterIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], latterrepresent);

	  offset = 4 * _poolfunctions;
    //after
    //avg pooling
    if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
      avgpool_forward(poolInput, pool[offset], poolIndex[offset], afterIndex);
    }
    //max pooling
    if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
      maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], afterIndex);
    }
    //min pooling
    if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
      minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], afterIndex);
    }
    //std pooling
    if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
      stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], afterIndex);
    }
    //pro pooling
    if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
      propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], afterIndex);
    }

    concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], afterrepresent);



  concat(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent, poolmerge);


    _olayer_linear.ComputeForwardScore(poolmerge, output);

    // decode algorithm
    int optLabel = softmax_predict(output, results);

    //release
    FreeSpace(&wordprime);
    FreeSpace(&wordrepresent);

    FreeSpace(&input);

    FreeSpace(&rnn_hidden_left_reset);
    FreeSpace(&rnn_hidden_left_update);
    FreeSpace(&rnn_hidden_left_afterreset);
    FreeSpace(&rnn_hidden_left_current);
    FreeSpace(&rnn_hidden_left);

    FreeSpace(&rnn_hidden_right_reset);
    FreeSpace(&rnn_hidden_right_update);
    FreeSpace(&rnn_hidden_right_afterreset);
    FreeSpace(&rnn_hidden_right_current);
    FreeSpace(&rnn_hidden_right);

    FreeSpace(&rnn_hidden_merge);

    FreeSpace(&project);


  	if((_channel & 2) == 2) {
  	}
  	if((_channel & 4) == 4) {
  	}
  	if((_channel & 8) == 8) {
        FreeSpace(&nerprime);
        FreeSpace(&ner_hidden_left_reset);
        FreeSpace(&ner_hidden_left_update);
        FreeSpace(&ner_hidden_left_afterreset);
        FreeSpace(&ner_hidden_left_current);
        FreeSpace(&ner_hidden_left);
        FreeSpace(&ner_hidden_right_reset);
        FreeSpace(&ner_hidden_right_update);
        FreeSpace(&ner_hidden_right_afterreset);
        FreeSpace(&ner_hidden_right_current);
        FreeSpace(&ner_hidden_right);
        FreeSpace(&ner_hidden_merge);
        FreeSpace(&ner_project);
  	}
  	if((_channel & 16) == 16) {
        FreeSpace(&posprime);
  	     FreeSpace(&pos_hidden_left_reset);
  	      FreeSpace(&pos_hidden_left_update);
  	      FreeSpace(&pos_hidden_left_afterreset);
  	      FreeSpace(&pos_hidden_left_current);
  	      FreeSpace(&pos_hidden_left);
  	      FreeSpace(&pos_hidden_right_reset);
  	      FreeSpace(&pos_hidden_right_update);
  	      FreeSpace(&pos_hidden_right_afterreset);
  	      FreeSpace(&pos_hidden_right_current);
  	      FreeSpace(&pos_hidden_right);
  	      FreeSpace(&pos_hidden_merge);
  	      FreeSpace(&pos_project);
  	}
  	if((_channel & 32) == 32) {
        FreeSpace(&sstprime);
        FreeSpace(&sst_hidden_left_reset);
        FreeSpace(&sst_hidden_left_update);
        FreeSpace(&sst_hidden_left_afterreset);
        FreeSpace(&sst_hidden_left_current);
        FreeSpace(&sst_hidden_left);
        FreeSpace(&sst_hidden_right_reset);
        FreeSpace(&sst_hidden_right_update);
        FreeSpace(&sst_hidden_right_afterreset);
        FreeSpace(&sst_hidden_right_current);
        FreeSpace(&sst_hidden_right);
        FreeSpace(&sst_hidden_merge);
        FreeSpace(&sst_project);
  	}

    FreeSpace(&poolInput);


    for (int idm = 0; idm < _poolmanners; idm++) {
      FreeSpace(&(pool[idm]));
      FreeSpace(&(poolIndex[idm]));
    }


    FreeSpace(&beforerepresent);
    FreeSpace(&formerrepresent);
    FreeSpace(&middlerepresent);
	  FreeSpace(&latterrepresent);
	  FreeSpace(&afterrepresent);

    FreeSpace(&poolmerge);
    FreeSpace(&output);

    return optLabel;
  }

  dtype computeScore(const Example& example) {
		const vector<Feature>& features = example.m_features;
  int seq_size = features.size();
  int offset = 0;

  Tensor<xpu, 3, dtype> input;
  Tensor<xpu, 3, dtype> project;

  Tensor<xpu, 3, dtype> rnn_hidden_left_update;
  Tensor<xpu, 3, dtype> rnn_hidden_left_reset;
  Tensor<xpu, 3, dtype> rnn_hidden_left;
  Tensor<xpu, 3, dtype> rnn_hidden_left_afterreset;
  Tensor<xpu, 3, dtype> rnn_hidden_left_current;

  Tensor<xpu, 3, dtype> rnn_hidden_right_update;
  Tensor<xpu, 3, dtype> rnn_hidden_right_reset;
  Tensor<xpu, 3, dtype> rnn_hidden_right;
  Tensor<xpu, 3, dtype> rnn_hidden_right_afterreset;
  Tensor<xpu, 3, dtype> rnn_hidden_right_current;

  Tensor<xpu, 3, dtype> rnn_hidden_merge;


  Tensor<xpu, 3, dtype> posprime;
  Tensor<xpu, 3, dtype> sstprime;
  Tensor<xpu, 3, dtype> nerprime;

  Tensor<xpu, 3, dtype> pos_hidden_left;
  Tensor<xpu, 3, dtype> pos_hidden_left_reset, pos_hidden_left_afterreset, pos_hidden_left_update, pos_hidden_left_current;
  Tensor<xpu, 3, dtype> pos_hidden_right;
  Tensor<xpu, 3, dtype> pos_hidden_right_reset, pos_hidden_right_afterreset, pos_hidden_right_update, pos_hidden_right_current;
  Tensor<xpu, 3, dtype> pos_hidden_merge;
  Tensor<xpu, 3, dtype> pos_project;

  Tensor<xpu, 3, dtype> sst_hidden_left;
  Tensor<xpu, 3, dtype> sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current;
  Tensor<xpu, 3, dtype> sst_hidden_right;
  Tensor<xpu, 3, dtype> sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current;
  Tensor<xpu, 3, dtype> sst_hidden_merge;
  Tensor<xpu, 3, dtype> sst_project;

  Tensor<xpu, 3, dtype> ner_hidden_left;
  Tensor<xpu, 3, dtype> ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current;
  Tensor<xpu, 3, dtype> ner_hidden_right;
  Tensor<xpu, 3, dtype> ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current;
  Tensor<xpu, 3, dtype> ner_hidden_merge;
  Tensor<xpu, 3, dtype> ner_project;

  Tensor<xpu, 3, dtype> poolInput;

  vector<Tensor<xpu, 2, dtype> > pool(_poolmanners);
  vector<Tensor<xpu, 3, dtype> > poolIndex(_poolmanners);

  Tensor<xpu, 2, dtype> poolmerge;
  Tensor<xpu, 2, dtype> output;


  Tensor<xpu, 3, dtype> wordprime, wordrepresent;

	hash_set<int> beforeIndex, formerIndex, middleIndex, latterIndex, afterIndex;
    Tensor<xpu, 2, dtype> beforerepresent;
    Tensor<xpu, 2, dtype> formerrepresent;
	  Tensor<xpu, 2, dtype> middlerepresent;
    Tensor<xpu, 2, dtype> latterrepresent;
	  Tensor<xpu, 2, dtype> afterrepresent;


  //initialize
  wordprime = NewTensor<xpu>(Shape3(seq_size, 1, _wordDim), 0.0);
  wordrepresent = NewTensor<xpu>(Shape3(seq_size, 1, _token_representation_size), 0.0);

  input = NewTensor<xpu>(Shape3(seq_size, 1, _inputsize), 0.0);

  rnn_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
  rnn_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
  rnn_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
  rnn_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
  rnn_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

  rnn_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
  rnn_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
  rnn_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
  rnn_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);
  rnn_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _rnnHiddenSize), 0.0);

  rnn_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_rnnHiddenSize), 0.0);

  project = NewTensor<xpu>(Shape3(seq_size, 1, _hiddensize), 0.0);


	if((_channel & 2) == 2) {
	}
	if((_channel & 4) == 4) {
	}
	if((_channel & 8) == 8) {
      nerprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      ner_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
      ner_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	}
	if((_channel & 16) == 16) {
      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      pos_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
      pos_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	}
	if((_channel & 32) == 32) {
      sstprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_left_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_left_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_left_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_left_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_left = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_right_reset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_right_update = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_right_afterreset = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_right_current = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_right = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      sst_hidden_merge = NewTensor<xpu>(Shape3(seq_size, 1, 2*_otherDim), 0.0);
      sst_project = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
	}

  poolInput = NewTensor<xpu>(Shape3(seq_size, 1, _poolInputSize), 0.0);

  for (int idm = 0; idm < _poolmanners; idm++) {
    pool[idm] = NewTensor<xpu>(Shape2(1, _poolInputSize), 0.0);
    poolIndex[idm] = NewTensor<xpu>(Shape3(seq_size, 1, _poolInputSize), 0.0);
  }

  beforerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
  formerrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
  middlerepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
	  latterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);
	  afterrepresent = NewTensor<xpu>(Shape2(1, _poolfunctions * _poolInputSize), 0.0);



  poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
  output = NewTensor<xpu>(Shape2(1, _labelSize), 0.0);

  //forward propagation
  //input setting, and linear setting
  for (int idx = 0; idx < seq_size; idx++) {
    const Feature& feature = features[idx];
    //linear features should not be dropped out

    const vector<int>& words = feature.words;
	    if (idx < example.formerTkBegin) {
      beforeIndex.insert(idx);
    } else if(idx >= example.formerTkBegin && idx <= example.formerTkEnd) {
			formerIndex.insert(idx);
		} else if (idx >= example.latterTkBegin && idx <= example.latterTkEnd) {
      latterIndex.insert(idx);
    } else if (idx > example.latterTkEnd) {
      afterIndex.insert(idx);
    } else {
      middleIndex.insert(idx);
    }

    _words.GetEmb(words[0], wordprime[idx]);

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
		   _ner.GetEmb(feature.ner, nerprime[idx]);
		}
		if((_channel & 16) == 16) {
		   _pos.GetEmb(feature.pos, posprime[idx]);
		}
		if((_channel & 32) == 32) {
		   _sst.GetEmb(feature.sst, sstprime[idx]);
		}

  }

  for (int idx = 0; idx < seq_size; idx++) {
    wordrepresent[idx] += wordprime[idx];
  }

  windowlized(wordrepresent, input, _wordcontext);

  _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
  _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

	if((_channel & 2) == 2) {
	}
	if((_channel & 4) == 4) {
	}
	if((_channel & 8) == 8) {
	      _ner_left.ComputeForwardScore(nerprime, ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current, ner_hidden_left);
	      _ner_right.ComputeForwardScore(nerprime, ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current, ner_hidden_right);
	}
	if((_channel & 16) == 16) {
	      _pos_left.ComputeForwardScore(posprime, pos_hidden_left_reset, pos_hidden_left_afterreset, pos_hidden_left_update, pos_hidden_left_current, pos_hidden_left);
	      _pos_right.ComputeForwardScore(posprime, pos_hidden_right_reset, pos_hidden_right_afterreset, pos_hidden_right_update, pos_hidden_right_current, pos_hidden_right);
	}
	if((_channel & 32) == 32) {
	      _sst_left.ComputeForwardScore(sstprime, sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current, sst_hidden_left);
	      _sst_right.ComputeForwardScore(sstprime, sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current, sst_hidden_right);
	}

  for (int idx = 0; idx < seq_size; idx++) {
      concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);

		if((_channel & 2) == 2) {
		}
		if((_channel & 4) == 4) {
		}
		if((_channel & 8) == 8) {
	        concat(ner_hidden_left[idx], ner_hidden_right[idx], ner_hidden_merge[idx]);
		}
		if((_channel & 16) == 16) {
	        concat(pos_hidden_left[idx], pos_hidden_right[idx], pos_hidden_merge[idx]);
		}
		if((_channel & 32) == 32) {
	        concat(sst_hidden_left[idx], sst_hidden_right[idx], sst_hidden_merge[idx]);
		}
  }

  // do we need a convolution? future work, currently needn't
  for (int idx = 0; idx < seq_size; idx++) {
    _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);

		if((_channel & 2) == 2) {
		}
		if((_channel & 4) == 4) {
		}
		if((_channel & 8) == 8) {
			_ner_project.ComputeForwardScore(ner_hidden_merge[idx], ner_project[idx]);
		}
		if((_channel & 16) == 16) {
			_pos_project.ComputeForwardScore(pos_hidden_merge[idx], pos_project[idx]);
		}
		if((_channel & 32) == 32) {
			_sst_project.ComputeForwardScore(sst_hidden_merge[idx], sst_project[idx]);
		}
  }


  for(int i=0;i<seq_size;i++) {
      vector<Tensor<xpu, 2, dtype> > v_otherInput;
      v_otherInput.push_back(project[i]);

		if((_channel & 2) == 2) {

		}
		if((_channel & 4) == 4) {

		}
		if((_channel & 8) == 8) {
			  v_otherInput.push_back(ner_project[i]);
		}
		if((_channel & 16) == 16) {
			  v_otherInput.push_back(pos_project[i]);
		}
		if((_channel & 32) == 32) {
			  v_otherInput.push_back(sst_project[i]);
		}

		concat(v_otherInput, poolInput[i]);
  }


  offset = 0;
  //before
  //avg pooling
  if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
    avgpool_forward(poolInput, pool[offset], poolIndex[offset], beforeIndex);
  }
  //max pooling
  if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
    maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], beforeIndex);
  }
  //min pooling
  if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
    minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], beforeIndex);
  }
  //std pooling
  if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
    stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], beforeIndex);
  }
  //pro pooling
  if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
    propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], beforeIndex);
  }

  concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], beforerepresent);

  offset = _poolfunctions;
  //former
  //avg pooling
  if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
    avgpool_forward(poolInput, pool[offset], poolIndex[offset], formerIndex);
  }
  //max pooling
  if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
    maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], formerIndex);
  }
  //min pooling
  if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
    minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], formerIndex);
  }
  //std pooling
  if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
    stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], formerIndex);
  }
  //pro pooling
  if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
    propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], formerIndex);
  }

  concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], formerrepresent);

  offset = 2 * _poolfunctions;
  //middle
  //avg pooling
  if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
    avgpool_forward(poolInput, pool[offset], poolIndex[offset], middleIndex);
  }
  //max pooling
  if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
    maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], middleIndex);
  }
  //min pooling
  if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
    minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], middleIndex);
  }
  //std pooling
  if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
    stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], middleIndex);
  }
  //pro pooling
  if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
    propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], middleIndex);
  }

  concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], middlerepresent);

	  offset = 3 * _poolfunctions;
  //latter
  //avg pooling
  if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
    avgpool_forward(poolInput, pool[offset], poolIndex[offset], latterIndex);
  }
  //max pooling
  if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
    maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], latterIndex);
  }
  //min pooling
  if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
    minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], latterIndex);
  }
  //std pooling
  if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
    stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], latterIndex);
  }
  //pro pooling
  if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
    propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], latterIndex);
  }

  concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], latterrepresent);

	  offset = 4 * _poolfunctions;
  //after
  //avg pooling
  if ((_remove > 0 && _remove != 1) || (_remove < 0 && _remove == -1) || _remove == 0) {
    avgpool_forward(poolInput, pool[offset], poolIndex[offset], afterIndex);
  }
  //max pooling
  if ((_remove > 0 && _remove != 2) || (_remove < 0 && _remove == -2) || _remove == 0) {
    maxpool_forward(poolInput, pool[offset + 1], poolIndex[offset + 1], afterIndex);
  }
  //min pooling
  if ((_remove > 0 && _remove != 3) || (_remove < 0 && _remove == -3) || _remove == 0) {
    minpool_forward(poolInput, pool[offset + 2], poolIndex[offset + 2], afterIndex);
  }
  //std pooling
  if ((_remove > 0 && _remove != 4) || (_remove < 0 && _remove == -4) || _remove == 0) {
    stdpool_forward(poolInput, pool[offset + 3], poolIndex[offset + 3], afterIndex);
  }
  //pro pooling
  if ((_remove > 0 && _remove != 5) || (_remove < 0 && _remove == -5) || _remove == 0) {
    propool_forward(poolInput, pool[offset + 4], poolIndex[offset + 4], afterIndex);
  }

  concat(pool[offset], pool[offset + 1], pool[offset + 2], pool[offset + 3], pool[offset + 4], afterrepresent);



concat(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent, poolmerge);


  _olayer_linear.ComputeForwardScore(poolmerge, output);

  // decode algorithm
  dtype cost = softmax_cost(output, example.m_labels);

  //release
  FreeSpace(&wordprime);
  FreeSpace(&wordrepresent);

  FreeSpace(&input);

  FreeSpace(&rnn_hidden_left_reset);
  FreeSpace(&rnn_hidden_left_update);
  FreeSpace(&rnn_hidden_left_afterreset);
  FreeSpace(&rnn_hidden_left_current);
  FreeSpace(&rnn_hidden_left);

  FreeSpace(&rnn_hidden_right_reset);
  FreeSpace(&rnn_hidden_right_update);
  FreeSpace(&rnn_hidden_right_afterreset);
  FreeSpace(&rnn_hidden_right_current);
  FreeSpace(&rnn_hidden_right);

  FreeSpace(&rnn_hidden_merge);

  FreeSpace(&project);


	if((_channel & 2) == 2) {
	}
	if((_channel & 4) == 4) {
	}
	if((_channel & 8) == 8) {
      FreeSpace(&nerprime);
      FreeSpace(&ner_hidden_left_reset);
      FreeSpace(&ner_hidden_left_update);
      FreeSpace(&ner_hidden_left_afterreset);
      FreeSpace(&ner_hidden_left_current);
      FreeSpace(&ner_hidden_left);
      FreeSpace(&ner_hidden_right_reset);
      FreeSpace(&ner_hidden_right_update);
      FreeSpace(&ner_hidden_right_afterreset);
      FreeSpace(&ner_hidden_right_current);
      FreeSpace(&ner_hidden_right);
      FreeSpace(&ner_hidden_merge);
      FreeSpace(&ner_project);
	}
	if((_channel & 16) == 16) {
      FreeSpace(&posprime);
	     FreeSpace(&pos_hidden_left_reset);
	      FreeSpace(&pos_hidden_left_update);
	      FreeSpace(&pos_hidden_left_afterreset);
	      FreeSpace(&pos_hidden_left_current);
	      FreeSpace(&pos_hidden_left);
	      FreeSpace(&pos_hidden_right_reset);
	      FreeSpace(&pos_hidden_right_update);
	      FreeSpace(&pos_hidden_right_afterreset);
	      FreeSpace(&pos_hidden_right_current);
	      FreeSpace(&pos_hidden_right);
	      FreeSpace(&pos_hidden_merge);
	      FreeSpace(&pos_project);
	}
	if((_channel & 32) == 32) {
      FreeSpace(&sstprime);
      FreeSpace(&sst_hidden_left_reset);
      FreeSpace(&sst_hidden_left_update);
      FreeSpace(&sst_hidden_left_afterreset);
      FreeSpace(&sst_hidden_left_current);
      FreeSpace(&sst_hidden_left);
      FreeSpace(&sst_hidden_right_reset);
      FreeSpace(&sst_hidden_right_update);
      FreeSpace(&sst_hidden_right_afterreset);
      FreeSpace(&sst_hidden_right_current);
      FreeSpace(&sst_hidden_right);
      FreeSpace(&sst_hidden_merge);
      FreeSpace(&sst_project);
	}

  FreeSpace(&poolInput);


  for (int idm = 0; idm < _poolmanners; idm++) {
    FreeSpace(&(pool[idm]));
    FreeSpace(&(poolIndex[idm]));
  }


  FreeSpace(&beforerepresent);
  FreeSpace(&formerrepresent);
  FreeSpace(&middlerepresent);
	  FreeSpace(&latterrepresent);
	  FreeSpace(&afterrepresent);

  FreeSpace(&poolmerge);
  FreeSpace(&output);

  return cost;
}

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _pos.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _ner.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sst.updateAdaGrad(nnRegular, adaAlpha, adaEps);

	if((_channel & 2) == 2) {
	}
	if((_channel & 4) == 4) {
	}
	if((_channel & 8) == 8) {
	    _ner_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _ner_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _ner_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}
	if((_channel & 16) == 16) {
	    _pos_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _pos_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _pos_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}
	if((_channel & 32) == 32) {
	    _sst_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _sst_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _sst_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	}
  }

  void writeModel();

  void loadModel();

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int idx = 0; idx < Wd.size(1); idx++)
      idCols.push_back(idx);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 3, dtype> Wd, Tensor<xpu, 3, dtype> gradWd, const string& mark, int iter) {
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols, idThirds;
    idRows.clear();
    idCols.clear();
    idThirds.clear();
    for (int i = 0; i < Wd.size(0); ++i)
      idRows.push_back(i);
    for (int i = 0; i < Wd.size(1); i++)
      idCols.push_back(i);
    for (int i = 0; i < Wd.size(2); i++)
      idThirds.push_back(i);

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());
    random_shuffle(idThirds.begin(), idThirds.end());

    int check_i = idRows[0], check_j = idCols[0], check_k = idThirds[0];

    dtype orginValue = Wd[check_i][check_j][check_k];

    Wd[check_i][check_j][check_k] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j][check_k] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j][check_k];

    printf("Iteration %d, Checking gradient for %s[%d][%d][%d]:\t", iter, mark.c_str(), check_i, check_j, check_k);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j][check_k] = orginValue;
  }

  void checkgrad(const vector<Example>& examples, Tensor<xpu, 2, dtype> Wd, Tensor<xpu, 2, dtype> gradWd, const string& mark, int iter,
      const hash_set<int>& indexes, bool bRow = true) {
    if(indexes.size() == 0) return;
    int charseed = mark.length();
    for (int i = 0; i < mark.length(); i++) {
      charseed = (int) (mark[i]) * 5 + charseed;
    }
    srand(iter + charseed);
    std::vector<int> idRows, idCols;
    idRows.clear();
    idCols.clear();
    static hash_set<int>::iterator it;
    if (bRow) {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idRows.push_back(*it);
      for (int idx = 0; idx < Wd.size(1); idx++)
        idCols.push_back(idx);
    } else {
      for (it = indexes.begin(); it != indexes.end(); ++it)
        idCols.push_back(*it);
      for (int idx = 0; idx < Wd.size(0); idx++)
        idRows.push_back(idx);
    }

    random_shuffle(idRows.begin(), idRows.end());
    random_shuffle(idCols.begin(), idCols.end());

    int check_i = idRows[0], check_j = idCols[0];

    dtype orginValue = Wd[check_i][check_j];

    Wd[check_i][check_j] = orginValue + 0.001;
    dtype lossAdd = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossAdd += computeScore(oneExam);
    }

    Wd[check_i][check_j] = orginValue - 0.001;
    dtype lossPlus = 0.0;
    for (int i = 0; i < examples.size(); i++) {
      Example oneExam = examples[i];
      lossPlus += computeScore(oneExam);
    }

    dtype mockGrad = (lossAdd - lossPlus) / 0.002;
    mockGrad = mockGrad / examples.size();
    dtype computeGrad = gradWd[check_i][check_j];

    printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter, mark.c_str(), check_i, check_j);
    printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

    Wd[check_i][check_j] = orginValue;

  }

  void checkgrads(const vector<Example>& examples, int iter) {


	    checkgrad(examples, _olayer_linear._W, _olayer_linear._gradW, "_olayer_linear._W", iter);
	    checkgrad(examples, _olayer_linear._b, _olayer_linear._gradb, "_olayer_linear._b", iter);

	    checkgrad(examples, _tanh_project._W, _tanh_project._gradW, "_tanh_project._W", iter);
	    checkgrad(examples, _tanh_project._b, _tanh_project._gradb, "_tanh_project._b", iter);

	    checkgrad(examples, _rnn_left._rnn_update._WL, _rnn_left._rnn_update._gradWL, "_rnn_left._rnn_update._WL", iter);
	    checkgrad(examples, _rnn_left._rnn_update._WR, _rnn_left._rnn_update._gradWR, "_rnn_left._rnn_update._WR", iter);
	    checkgrad(examples, _rnn_left._rnn_update._b, _rnn_left._rnn_update._gradb, "_rnn_left._rnn_update._b", iter);
	    checkgrad(examples, _rnn_left._rnn_reset._WL, _rnn_left._rnn_reset._gradWL, "_rnn_left._rnn_reset._WL", iter);
	    checkgrad(examples, _rnn_left._rnn_reset._WR, _rnn_left._rnn_reset._gradWR, "_rnn_left._rnn_reset._WR", iter);
	    checkgrad(examples, _rnn_left._rnn_reset._b, _rnn_left._rnn_reset._gradb, "_rnn_left._rnn_reset._b", iter);
	    checkgrad(examples, _rnn_left._rnn._WL, _rnn_left._rnn._gradWL, "_rnn_left._rnn._WL", iter);
	    checkgrad(examples, _rnn_left._rnn._WR, _rnn_left._rnn._gradWR, "_rnn_left._rnn._WR", iter);
	    checkgrad(examples, _rnn_left._rnn._b, _rnn_left._rnn._gradb, "_rnn_left._rnn._b", iter);

	    checkgrad(examples, _rnn_right._rnn_update._WL, _rnn_right._rnn_update._gradWL, "_rnn_right._rnn_update._WL", iter);
	    checkgrad(examples, _rnn_right._rnn_update._WR, _rnn_right._rnn_update._gradWR, "_rnn_right._rnn_update._WR", iter);
	    checkgrad(examples, _rnn_right._rnn_update._b, _rnn_right._rnn_update._gradb, "_rnn_right._rnn_update._b", iter);
	    checkgrad(examples, _rnn_right._rnn_reset._WL, _rnn_right._rnn_reset._gradWL, "_rnn_right._rnn_reset._WL", iter);
	    checkgrad(examples, _rnn_right._rnn_reset._WR, _rnn_right._rnn_reset._gradWR, "_rnn_right._rnn_reset._WR", iter);
	    checkgrad(examples, _rnn_right._rnn_reset._b, _rnn_right._rnn_reset._gradb, "_rnn_right._rnn_reset._b", iter);
	    checkgrad(examples, _rnn_right._rnn._WL, _rnn_right._rnn._gradWL, "_rnn_right._rnn._WL", iter);
	    checkgrad(examples, _rnn_right._rnn._WR, _rnn_right._rnn._gradWR, "_rnn_right._rnn._WR", iter);
	    checkgrad(examples, _rnn_right._rnn._b, _rnn_right._rnn._gradb, "_rnn_right._rnn._b", iter);

	    checkgrad(examples, _ner_project._W, _ner_project._gradW, "_ner_project._W", iter);
	    checkgrad(examples, _ner_project._b, _ner_project._gradb, "_ner_project._b", iter);
	    checkgrad(examples, _ner_left._rnn_update._WL, _ner_left._rnn_update._gradWL, "_ner_left._rnn_update._WL", iter);
	    checkgrad(examples, _ner_left._rnn_update._WR, _ner_left._rnn_update._gradWR, "_ner_left._rnn_update._WR", iter);
	    checkgrad(examples, _ner_left._rnn_update._b, _ner_left._rnn_update._gradb, "_ner_left._rnn_update._b", iter);
	    checkgrad(examples, _ner_left._rnn_reset._WL, _ner_left._rnn_reset._gradWL, "_ner_left._rnn_reset._WL", iter);
	    checkgrad(examples, _ner_left._rnn_reset._WR, _ner_left._rnn_reset._gradWR, "_ner_left._rnn_reset._WR", iter);
	    checkgrad(examples, _ner_left._rnn_reset._b, _ner_left._rnn_reset._gradb, "_ner_left._rnn_reset._b", iter);
	    checkgrad(examples, _ner_left._rnn._WL, _ner_left._rnn._gradWL, "_ner_left._rnn._WL", iter);
	    checkgrad(examples, _ner_left._rnn._WR, _ner_left._rnn._gradWR, "_ner_left._rnn._WR", iter);
	    checkgrad(examples, _ner_left._rnn._b, _ner_left._rnn._gradb, "_ner_left._rnn._b", iter);
	    checkgrad(examples, _ner_right._rnn_update._WL, _ner_right._rnn_update._gradWL, "_ner_right._rnn_update._WL", iter);
	    checkgrad(examples, _ner_right._rnn_update._WR, _ner_right._rnn_update._gradWR, "_ner_right._rnn_update._WR", iter);
	    checkgrad(examples, _ner_right._rnn_update._b, _ner_right._rnn_update._gradb, "_ner_right._rnn_update._b", iter);
	    checkgrad(examples, _ner_right._rnn_reset._WL, _ner_right._rnn_reset._gradWL, "_ner_right._rnn_reset._WL", iter);
	    checkgrad(examples, _ner_right._rnn_reset._WR, _ner_right._rnn_reset._gradWR, "_ner_right._rnn_reset._WR", iter);
	    checkgrad(examples, _ner_right._rnn_reset._b, _ner_right._rnn_reset._gradb, "_ner_right._rnn_reset._b", iter);
	    checkgrad(examples, _ner_right._rnn._WL, _ner_right._rnn._gradWL, "_ner_right._rnn._WL", iter);
	    checkgrad(examples, _ner_right._rnn._WR, _ner_right._rnn._gradWR, "_ner_right._rnn._WR", iter);
	    checkgrad(examples, _ner_right._rnn._b, _ner_right._rnn._gradb, "_ner_right._rnn._b", iter);

	    checkgrad(examples, _pos_project._W, _pos_project._gradW, "_pos_project._W", iter);
	    checkgrad(examples, _pos_project._b, _pos_project._gradb, "_pos_project._b", iter);
	    checkgrad(examples, _pos_left._rnn_update._WL, _pos_left._rnn_update._gradWL, "_pos_left._rnn_update._WL", iter);
	    checkgrad(examples, _pos_left._rnn_update._WR, _pos_left._rnn_update._gradWR, "_pos_left._rnn_update._WR", iter);
	    checkgrad(examples, _pos_left._rnn_update._b, _pos_left._rnn_update._gradb, "_pos_left._rnn_update._b", iter);
	    checkgrad(examples, _pos_left._rnn_reset._WL, _pos_left._rnn_reset._gradWL, "_pos_left._rnn_reset._WL", iter);
	    checkgrad(examples, _pos_left._rnn_reset._WR, _pos_left._rnn_reset._gradWR, "_pos_left._rnn_reset._WR", iter);
	    checkgrad(examples, _pos_left._rnn_reset._b, _pos_left._rnn_reset._gradb, "_pos_left._rnn_reset._b", iter);
	    checkgrad(examples, _pos_left._rnn._WL, _pos_left._rnn._gradWL, "_pos_left._rnn._WL", iter);
	    checkgrad(examples, _pos_left._rnn._WR, _pos_left._rnn._gradWR, "_pos_left._rnn._WR", iter);
	    checkgrad(examples, _pos_left._rnn._b, _pos_left._rnn._gradb, "_pos_left._rnn._b", iter);
	    checkgrad(examples, _pos_right._rnn_update._WL, _pos_right._rnn_update._gradWL, "_pos_right._rnn_update._WL", iter);
	    checkgrad(examples, _pos_right._rnn_update._WR, _pos_right._rnn_update._gradWR, "_pos_right._rnn_update._WR", iter);
	    checkgrad(examples, _pos_right._rnn_update._b, _pos_right._rnn_update._gradb, "_pos_right._rnn_update._b", iter);
	    checkgrad(examples, _pos_right._rnn_reset._WL, _pos_right._rnn_reset._gradWL, "_pos_right._rnn_reset._WL", iter);
	    checkgrad(examples, _pos_right._rnn_reset._WR, _pos_right._rnn_reset._gradWR, "_pos_right._rnn_reset._WR", iter);
	    checkgrad(examples, _pos_right._rnn_reset._b, _pos_right._rnn_reset._gradb, "_pos_right._rnn_reset._b", iter);
	    checkgrad(examples, _pos_right._rnn._WL, _pos_right._rnn._gradWL, "_pos_right._rnn._WL", iter);
	    checkgrad(examples, _pos_right._rnn._WR, _pos_right._rnn._gradWR, "_pos_right._rnn._WR", iter);
	    checkgrad(examples, _pos_right._rnn._b, _pos_right._rnn._gradb, "_pos_right._rnn._b", iter);

	    checkgrad(examples, _sst_project._W, _sst_project._gradW, "_sst_project._W", iter);
	    checkgrad(examples, _sst_project._b, _sst_project._gradb, "_sst_project._b", iter);
	    checkgrad(examples, _sst_left._rnn_update._WL, _sst_left._rnn_update._gradWL, "_sst_left._rnn_update._WL", iter);
	    checkgrad(examples, _sst_left._rnn_update._WR, _sst_left._rnn_update._gradWR, "_sst_left._rnn_update._WR", iter);
	    checkgrad(examples, _sst_left._rnn_update._b, _sst_left._rnn_update._gradb, "_sst_left._rnn_update._b", iter);
	    checkgrad(examples, _sst_left._rnn_reset._WL, _sst_left._rnn_reset._gradWL, "_sst_left._rnn_reset._WL", iter);
	    checkgrad(examples, _sst_left._rnn_reset._WR, _sst_left._rnn_reset._gradWR, "_sst_left._rnn_reset._WR", iter);
	    checkgrad(examples, _sst_left._rnn_reset._b, _sst_left._rnn_reset._gradb, "_sst_left._rnn_reset._b", iter);
	    checkgrad(examples, _sst_left._rnn._WL, _sst_left._rnn._gradWL, "_sst_left._rnn._WL", iter);
	    checkgrad(examples, _sst_left._rnn._WR, _sst_left._rnn._gradWR, "_sst_left._rnn._WR", iter);
	    checkgrad(examples, _sst_left._rnn._b, _sst_left._rnn._gradb, "_sst_left._rnn._b", iter);
	    checkgrad(examples, _sst_right._rnn_update._WL, _sst_right._rnn_update._gradWL, "_sst_right._rnn_update._WL", iter);
	    checkgrad(examples, _sst_right._rnn_update._WR, _sst_right._rnn_update._gradWR, "_sst_right._rnn_update._WR", iter);
	    checkgrad(examples, _sst_right._rnn_update._b, _sst_right._rnn_update._gradb, "_sst_right._rnn_update._b", iter);
	    checkgrad(examples, _sst_right._rnn_reset._WL, _sst_right._rnn_reset._gradWL, "_sst_right._rnn_reset._WL", iter);
	    checkgrad(examples, _sst_right._rnn_reset._WR, _sst_right._rnn_reset._gradWR, "_sst_right._rnn_reset._WR", iter);
	    checkgrad(examples, _sst_right._rnn_reset._b, _sst_right._rnn_reset._gradb, "_sst_right._rnn_reset._b", iter);
	    checkgrad(examples, _sst_right._rnn._WL, _sst_right._rnn._gradWL, "_sst_right._rnn._WL", iter);
	    checkgrad(examples, _sst_right._rnn._WR, _sst_right._rnn._gradWR, "_sst_right._rnn._WR", iter);
	    checkgrad(examples, _sst_right._rnn._b, _sst_right._rnn._gradb, "_sst_right._rnn._b", iter);


	    checkgrad(examples, _words._E, _words._gradE, "_words._E", iter, _words._indexers);
	    checkgrad(examples, _pos._E, _pos._gradE, "_pos._E", iter, _pos._indexers);
	    checkgrad(examples, _ner._E, _ner._gradE, "_ner._E", iter, _ner._indexers);
	    checkgrad(examples, _sst._E, _sst._gradE, "_sst._E", iter, _sst._indexers);

  }

public:
  inline void resetEval() {
    _eval.reset();
  }

  inline void setDropValue(dtype dropOut) {
    _dropOut = dropOut;
  }

  inline void setWordEmbFinetune(bool b_wordEmb_finetune) {
    _words.setEmbFineTune(b_wordEmb_finetune);
  }

  inline void resetRemove(int remove) {
    _remove = remove;
  }
};

#endif /* SRC_PoolGRNNClassifier_H_ */
