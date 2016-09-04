
#ifndef SRC_PoolExGRNNandRecursiveNNClassifier7_H_
#define SRC_PoolExGRNNandRecursiveNNClassifier7_H_

#include <iostream>

#include <assert.h>
#include "Att2GatedNN.h"
#include "Example.h"
#include "Feature.h"
#include "N3L.h"
#include "SemiDepRecursiveNN.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

//A native neural network classfier using only word embeddings
template<typename xpu>
class PoolExGRNNandRecursiveNNClassifier7 {
public:
	PoolExGRNNandRecursiveNNClassifier7() {
    _dropOut = 0.5;
  }
  ~PoolExGRNNandRecursiveNNClassifier7() {

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

  UniLayer<xpu> _represent_transform[5];
  Att2GatedNN<xpu> _target_attention;

  UniLayer<xpu> _olayer_linear;
  UniLayer<xpu> _tanh_project;

  GRNN<xpu> _rnn_left;
  GRNN<xpu> _rnn_right;

  GRNN<xpu> _sst_left;
  GRNN<xpu> _sst_right;
  UniLayer<xpu> _sst_project;
  GRNN<xpu> _ner_left;
  GRNN<xpu> _ner_right;
  UniLayer<xpu> _ner_project;


  SemiDepRecursiveNN<xpu> _recursive;

  int _poolmanners;
  int _poolfunctions;
  int _targetdim;

  int _poolsize;
  int _gatedsize;

  int _labelSize;

  Metric _eval;

  dtype _dropOut;

  int _remove; // 1, avg, 2, max, 3, min, 4, std, 5, pro

  Options options;

  int _wordWindowSize;
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

    _channel = options.channelMode;
    _otherDim = options.otherEmbSize;

    _wordWindowSize = _wordwindow * _token_representation_size;
    _inputsize = _wordwindow * _token_representation_size + _otherDim;
    _hiddensize = options.wordEmbSize;
    _rnnHiddenSize = options.rnnHiddenSize;

    _otherInputSize = 0;
	if((_channel & 8) == 8) {
		_otherInputSize += _otherDim;
	}
	if((_channel & 32) == 32) {
		_otherInputSize += _otherDim;
	}

    _poolInputSize = 2*_hiddensize + _otherInputSize;
    _poolsize = _poolmanners * _poolInputSize;


    _targetdim = _hiddensize;
    _gatedsize = _targetdim;

    _words.initial(wordEmb);

    for (int idx = 0; idx < 5; idx++) {
      _represent_transform[idx].initial(_targetdim, _poolfunctions * _poolInputSize, true, (idx + 1) * 100 + 60, 0);
    }
    _target_attention.initial(_targetdim, _targetdim, 100);

    _rnn_left.initial(_rnnHiddenSize, _inputsize, true, 10);
    _rnn_right.initial(_rnnHiddenSize, _inputsize, false, 40);


	if((_channel & 8) == 8) {
	    _ner_left.initial(_otherDim, _otherDim, true, 210);
	    _ner_right.initial(_otherDim, _otherDim, false, 220);
	    _ner_project.initial(_otherDim, 2*_otherDim, true, 230, 0);
	}
	if((_channel & 32) == 32) {
	    _sst_left.initial(_otherDim, _otherDim, true, 270);
	    _sst_right.initial(_otherDim, _otherDim, false, 280);
	    _sst_project.initial(_otherDim, 2*_otherDim, true, 290, 0);
	}

    _recursive.initial(_hiddensize, _inputsize, 20);

    _tanh_project.initial(_hiddensize, 2*_rnnHiddenSize, true, 70, 0);
    _olayer_linear.initial(_labelSize, _poolsize + _gatedsize , false, 80, 2);

    _remove = 0;

    cout<<"PoolExGRNNandRecursiveNNClassifier7 initial"<<endl;
    cout<< "Att2GatedNN, SemiDep" <<endl;
    cout<<"add POS tags to GRNN input, channelmode should be set"<<endl;
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


	if((_channel & 8) == 8) {
	    _ner_left.release();
	    _ner_right.release();
	    _ner_project.release();
	}
	if((_channel & 32) == 32) {
	    _sst_left.release();
	    _sst_right.release();
	    _sst_project.release();
	}

    for (int idx = 0; idx < 5; idx++) {
      _represent_transform[idx].release();
    }
    _target_attention.release();
    _recursive.release();

  }

  inline dtype process(const vector<Example>& examples, int iter) {
    _eval.reset();

    int example_num = examples.size();
    dtype cost = 0.0;
    int offset = 0;
    for (int count = 0; count < example_num; count++) {
      const Example& example = examples[count];

      int seq_size = example.m_features.size();
      int sentNum = example.dep.size();

      Tensor<xpu, 3, dtype> input, inputLoss;
      Tensor<xpu, 3, dtype> project, projectLoss;

      Tensor<xpu, 3, dtype> rnn_hidden_left, rnn_hidden_leftLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current;
      Tensor<xpu, 3, dtype> rnn_hidden_right, rnn_hidden_rightLoss;
      Tensor<xpu, 3, dtype> rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current;

      Tensor<xpu, 3, dtype> rnn_hidden_merge, rnn_hidden_mergeLoss;

      vector< Tensor<xpu, 3, dtype> > v_recursive_input(sentNum), v_recursive_input_loss(sentNum);
      vector< Tensor<xpu, 3, dtype> > v_recursive_rsp(sentNum);
      vector< vector< Tensor<xpu, 3, dtype> > > v_recursive_v_rsc(sentNum);
      vector< Tensor<xpu, 3, dtype> > v_recursive_hidden(sentNum), v_recursive_hidden_loss(sentNum);

      Tensor<xpu, 3, dtype> posprime, posprimeLoss, posprimeMask;
      Tensor<xpu, 3, dtype> wordWindow, wordWindowLoss;
      Tensor<xpu, 3, dtype> sstprime, sstprimeLoss, sstprimeMask;
      Tensor<xpu, 3, dtype> nerprime, nerprimeLoss, nerprimeMask;

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
      Tensor<xpu, 2, dtype> gatedmerge, gatedmergeLoss;
      Tensor<xpu, 2, dtype> allmerge, allmergeLoss;
      Tensor<xpu, 2, dtype> output, outputLoss;

      //gated interaction part
      Tensor<xpu, 2, dtype> input_span[5], input_spanLoss[5];
      Tensor<xpu, 2, dtype> reset_before, reset_middle, reset_after;
      Tensor<xpu, 2, dtype> interact, interactLoss;


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

      for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
    	  int sentLength = sentIdx==0 ? example.sentEnd[sentIdx] : example.sentEnd[sentIdx]-example.sentEnd[sentIdx-1];
    	  v_recursive_input[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _inputsize), 0.0);
    	  v_recursive_input_loss[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _inputsize), 0.0);
    	  v_recursive_rsp[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _hiddensize), 0.0);
    	  v_recursive_hidden[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _hiddensize), 0.0);
    	  v_recursive_hidden_loss[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _hiddensize), 0.0);
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

        posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        posprimeLoss = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        posprimeMask = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 1.0);
        wordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _wordWindowSize), 0.0);
        wordWindowLoss = NewTensor<xpu>(Shape3(seq_size, 1, _wordWindowSize), 0.0);

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

      for (int idm = 0; idm < 5; idm++) {
        input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
        input_spanLoss[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      }
      reset_before = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      reset_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      reset_after = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      interactLoss = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);


      poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      poolmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
      gatedmerge = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      gatedmergeLoss = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
      allmerge = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize ), 0.0);
      allmergeLoss = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize ), 0.0);
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

		if((_channel & 8) == 8) {
		   _ner.GetEmb(feature.ner, nerprime[idx]);
			//dropout
			dropoutcol(nerprimeMask[idx], _dropOut);
			nerprime[idx] = nerprime[idx] * nerprimeMask[idx];
		}

		   _pos.GetEmb(feature.pos, posprime[idx]);
			//dropout
			dropoutcol(posprimeMask[idx], _dropOut);
			posprime[idx] = posprime[idx] * posprimeMask[idx];

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

      windowlized(wordrepresent, wordWindow, _wordcontext);
      for (int idx = 0; idx < seq_size; idx++) {
    	  concat(wordWindow[idx], posprime[idx], input[idx]);
      }

      _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
      _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);

		if((_channel & 8) == 8) {
		      _ner_left.ComputeForwardScore(nerprime, ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current, ner_hidden_left);
		      _ner_right.ComputeForwardScore(nerprime, ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current, ner_hidden_right);
		}
		if((_channel & 32) == 32) {
		      _sst_left.ComputeForwardScore(sstprime, sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current, sst_hidden_left);
		      _sst_right.ComputeForwardScore(sstprime, sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current, sst_hidden_right);
		}

      for (int idx = 0; idx < seq_size; idx++) {
        concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);

		if((_channel & 8) == 8) {
	        concat(ner_hidden_left[idx], ner_hidden_right[idx], ner_hidden_merge[idx]);
		}
		if((_channel & 32) == 32) {
	        concat(sst_hidden_left[idx], sst_hidden_right[idx], sst_hidden_merge[idx]);
		}
      }

      // do we need a convolution? future work, currently needn't
      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);

		if((_channel & 8) == 8) {
			_ner_project.ComputeForwardScore(ner_hidden_merge[idx], ner_project[idx]);
		}
		if((_channel & 32) == 32) {
			_sst_project.ComputeForwardScore(sst_hidden_merge[idx], sst_project[idx]);
		}
      }


      int sentBegin = 0;
      for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
    	  int sentLength = sentIdx==0 ? example.sentEnd[sentIdx] : example.sentEnd[sentIdx]-example.sentEnd[sentIdx-1];
    	  int seqIdx = sentBegin;
    	  for(int idx=0;idx<sentLength;idx++, seqIdx++) {
    		  v_recursive_input[sentIdx][idx] += input[seqIdx];
    	  }

          _recursive.ComputeForwardScore(v_recursive_input[sentIdx], example.dep[sentIdx], example.depType[sentIdx],
        		  v_recursive_v_rsc[sentIdx], v_recursive_rsp[sentIdx],
        		  v_recursive_hidden[sentIdx]);

          seqIdx = sentBegin;
    	  for(int idx=0;idx<sentLength;idx++, seqIdx++) {
              vector<Tensor<xpu, 2, dtype> > v_otherInput;
              v_otherInput.push_back(project[seqIdx]);
              v_otherInput.push_back(v_recursive_hidden[sentIdx][idx]);


        		if((_channel & 8) == 8) {
        			  v_otherInput.push_back(ner_project[seqIdx]);
        		}
        		if((_channel & 32) == 32) {
        			  v_otherInput.push_back(sst_project[seqIdx]);
        		}

        		concat(v_otherInput, poolInput[seqIdx]);

    	  }

    	  sentBegin += sentLength;
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

      _represent_transform[0].ComputeForwardScore(beforerepresent, input_span[0]);
      _represent_transform[1].ComputeForwardScore(formerrepresent, input_span[1]);
      _represent_transform[2].ComputeForwardScore(middlerepresent, input_span[2]);
      _represent_transform[3].ComputeForwardScore(latterrepresent, input_span[3]);
      _represent_transform[4].ComputeForwardScore(afterrepresent, input_span[4]);

      _target_attention.ComputeForwardScore(input_span[0], input_span[2], input_span[4],
    		  input_span[1], input_span[3],
            reset_before, reset_middle, reset_after,
			interact);

      concat(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent, poolmerge);
      gatedmerge += interact;

      concat(poolmerge, gatedmerge, allmerge);

      _olayer_linear.ComputeForwardScore(allmerge, output);

      // get delta for each output
      cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

      // loss backward propagation
      _olayer_linear.ComputeBackwardLoss(allmerge, output, outputLoss, allmergeLoss);


      unconcat(poolmergeLoss, gatedmergeLoss, allmergeLoss);

      interactLoss += gatedmergeLoss;
      unconcat(beforerepresentLoss, formerrepresentLoss, middlerepresentLoss, latterrepresentLoss, afterrepresentLoss, poolmergeLoss);

      _target_attention.ComputeBackwardLoss(input_span[0], input_span[2], input_span[4],
    		  input_span[1], input_span[3],
			  reset_before, reset_middle, reset_after,
			  interact, interactLoss,
			  input_spanLoss[0], input_spanLoss[2], input_spanLoss[4],
			  input_spanLoss[1], input_spanLoss[3]);

      _represent_transform[0].ComputeBackwardLoss(beforerepresent, input_span[0], input_spanLoss[0], beforerepresentLoss);
      _represent_transform[1].ComputeBackwardLoss(formerrepresent, input_span[1], input_spanLoss[1], formerrepresentLoss);
      _represent_transform[2].ComputeBackwardLoss(middlerepresent, input_span[2], input_spanLoss[2], middlerepresentLoss);
      _represent_transform[3].ComputeBackwardLoss(latterrepresent, input_span[3], input_spanLoss[3], latterrepresentLoss);
      _represent_transform[4].ComputeBackwardLoss(afterrepresent, input_span[4], input_spanLoss[4], afterrepresentLoss);


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

      sentBegin = 0;
      for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
    	  int sentLength = sentIdx==0 ? example.sentEnd[sentIdx] : example.sentEnd[sentIdx]-example.sentEnd[sentIdx-1];
    	  int seqIdx = sentBegin;
    	  for(int idx=0;idx<sentLength;idx++, seqIdx++) {
        	  vector<Tensor<xpu, 2, dtype> > v_otherInputLoss;
        	  v_otherInputLoss.push_back(projectLoss[seqIdx]);
        	  v_otherInputLoss.push_back(v_recursive_hidden_loss[sentIdx][idx]);

        		if((_channel & 8) == 8) {
        			v_otherInputLoss.push_back(ner_projectLoss[seqIdx]);
        		}
        		if((_channel & 32) == 32) {
        			v_otherInputLoss.push_back(sst_projectLoss[seqIdx]);
        		}

        		unconcat(v_otherInputLoss, poolInputLoss[seqIdx]);

    	  }

          _recursive.ComputeBackwardLoss(v_recursive_input[sentIdx], example.dep[sentIdx], example.depType[sentIdx],
        		  v_recursive_v_rsc[sentIdx], v_recursive_rsp[sentIdx],
				  v_recursive_hidden[sentIdx], v_recursive_hidden_loss[sentIdx], v_recursive_input_loss[sentIdx]);


          seqIdx = sentBegin;
    	  for(int idx=0;idx<sentLength;idx++, seqIdx++) {
    		  inputLoss[seqIdx] += v_recursive_input_loss[sentIdx][idx];
    	  }

    	  sentBegin += sentLength;
      }



      for (int idx = 0; idx < seq_size; idx++) {
        _tanh_project.ComputeBackwardLoss(rnn_hidden_merge[idx], project[idx], projectLoss[idx], rnn_hidden_mergeLoss[idx]);

    	if((_channel & 8) == 8) {
            _ner_project.ComputeBackwardLoss(ner_hidden_merge[idx], ner_project[idx], ner_projectLoss[idx], ner_hidden_mergeLoss[idx]);
    	}
    	if((_channel & 32) == 32) {
            _sst_project.ComputeBackwardLoss(sst_hidden_merge[idx], sst_project[idx], sst_projectLoss[idx], sst_hidden_mergeLoss[idx]);
    	}
      }

      for (int idx = 0; idx < seq_size; idx++) {
        unconcat(rnn_hidden_leftLoss[idx], rnn_hidden_rightLoss[idx], rnn_hidden_mergeLoss[idx]);

    	if((_channel & 8) == 8) {
    		unconcat(ner_hidden_leftLoss[idx], ner_hidden_rightLoss[idx], ner_hidden_mergeLoss[idx]);
    	}
    	if((_channel & 32) == 32) {
    		unconcat(sst_hidden_leftLoss[idx], sst_hidden_rightLoss[idx], sst_hidden_mergeLoss[idx]);
    	}
      }

      _rnn_left.ComputeBackwardLoss(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left, rnn_hidden_leftLoss, inputLoss);
      _rnn_right.ComputeBackwardLoss(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right, rnn_hidden_rightLoss, inputLoss);

  	if((_channel & 8) == 8) {
        _ner_left.ComputeBackwardLoss(nerprime, ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current, ner_hidden_left, ner_hidden_leftLoss, nerprimeLoss);
        _ner_right.ComputeBackwardLoss(nerprime, ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current, ner_hidden_right, ner_hidden_rightLoss, nerprimeLoss);
  	}
  	if((_channel & 32) == 32) {
        _sst_left.ComputeBackwardLoss(sstprime, sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current, sst_hidden_left, sst_hidden_leftLoss, sstprimeLoss);
        _sst_right.ComputeBackwardLoss(sstprime, sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current, sst_hidden_right, sst_hidden_rightLoss, sstprimeLoss);
  	}


    for (int idx = 0; idx < seq_size; idx++) {
  	  unconcat(wordWindowLoss[idx], posprimeLoss[idx], inputLoss[idx]);
    }
    // word context
    windowlized_backward(wordrepresentLoss, wordWindowLoss, _wordcontext);

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

  		if((_channel & 8) == 8) {
  	          nerprimeLoss[idx] = nerprimeLoss[idx] * nerprimeMask[idx];
  			_ner.EmbLoss(feature.ner, nerprimeLoss[idx]);
  		}

	          posprimeLoss[idx] = posprimeLoss[idx] * posprimeMask[idx];
			_pos.EmbLoss(feature.pos, posprimeLoss[idx]);

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

        FreeSpace(&posprime);
        FreeSpace(&posprimeLoss);
        FreeSpace(&posprimeMask);
        FreeSpace(&wordWindow);
        FreeSpace(&wordWindowLoss);

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


    for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
  	  FreeSpace(&(v_recursive_input[sentIdx]));
  	  FreeSpace(&(v_recursive_input_loss[sentIdx]));
  	  FreeSpace(&(v_recursive_rsp[sentIdx]));
        for(int i=0;i<v_recursive_v_rsc[sentIdx].size();i++)
      	  FreeSpace(&(v_recursive_v_rsc[sentIdx][i]));
  	  FreeSpace(&(v_recursive_hidden[sentIdx]));
  	  FreeSpace(&(v_recursive_hidden_loss[sentIdx]));
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

      for (int idm = 0; idm < 5; idm++) {
        FreeSpace(&(input_span[idm]));
        FreeSpace(&(input_spanLoss[idm]));
      }
      FreeSpace(&reset_before);
      FreeSpace(&reset_middle);
      FreeSpace(&reset_after);
      FreeSpace(&(interact));
      FreeSpace(&(interactLoss));


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
      FreeSpace(&gatedmerge);
      FreeSpace(&gatedmergeLoss);
      FreeSpace(&allmerge);
      FreeSpace(&allmergeLoss);
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
    int sentNum = example.dep.size();

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

    vector< Tensor<xpu, 3, dtype> > v_recursive_input(sentNum);
    vector< Tensor<xpu, 3, dtype> > v_recursive_rsp(sentNum);
    vector< vector< Tensor<xpu, 3, dtype> > > v_recursive_v_rsc(sentNum);
    vector< Tensor<xpu, 3, dtype> > v_recursive_hidden(sentNum);

    Tensor<xpu, 3, dtype> posprime;
    Tensor<xpu, 3, dtype> wordWindow;
    Tensor<xpu, 3, dtype> sstprime;
    Tensor<xpu, 3, dtype> nerprime;

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
    Tensor<xpu, 2, dtype> gatedmerge;
    Tensor<xpu, 2, dtype> allmerge;
    Tensor<xpu, 2, dtype> output;

    //gated interaction part
    Tensor<xpu, 2, dtype> input_span[5];
    Tensor<xpu, 2, dtype> reset_before, reset_middle, reset_after;
    Tensor<xpu, 2, dtype> interact;

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

    for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
  	  int sentLength = sentIdx==0 ? example.sentEnd[sentIdx] : example.sentEnd[sentIdx]-example.sentEnd[sentIdx-1];
  	  v_recursive_input[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _inputsize), 0.0);
  	  v_recursive_rsp[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _hiddensize), 0.0);
  	  v_recursive_hidden[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _hiddensize), 0.0);
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

        posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
        wordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _wordWindowSize), 0.0);

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

      for (int idm = 0; idm < 5; idm++) {
        input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      }
      reset_before = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      reset_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      reset_after = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
      interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);


    poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
    gatedmerge = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
    allmerge = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize ), 0.0);
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

		if((_channel & 8) == 8) {
		   _ner.GetEmb(feature.ner, nerprime[idx]);
		}

		   _pos.GetEmb(feature.pos, posprime[idx]);

		if((_channel & 32) == 32) {
		   _sst.GetEmb(feature.sst, sstprime[idx]);
		}

    }

    for (int idx = 0; idx < seq_size; idx++) {
      wordrepresent[idx] += wordprime[idx];
    }

    windowlized(wordrepresent, wordWindow, _wordcontext);
    for (int idx = 0; idx < seq_size; idx++) {
    	concat(wordWindow[idx], posprime[idx], input[idx]);
    }

    _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
    _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);


	if((_channel & 8) == 8) {
	      _ner_left.ComputeForwardScore(nerprime, ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current, ner_hidden_left);
	      _ner_right.ComputeForwardScore(nerprime, ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current, ner_hidden_right);
	}
	if((_channel & 32) == 32) {
	      _sst_left.ComputeForwardScore(sstprime, sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current, sst_hidden_left);
	      _sst_right.ComputeForwardScore(sstprime, sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current, sst_hidden_right);
	}

    for (int idx = 0; idx < seq_size; idx++) {
        concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);

		if((_channel & 8) == 8) {
	        concat(ner_hidden_left[idx], ner_hidden_right[idx], ner_hidden_merge[idx]);
		}
		if((_channel & 32) == 32) {
	        concat(sst_hidden_left[idx], sst_hidden_right[idx], sst_hidden_merge[idx]);
		}
    }

    // do we need a convolution? future work, currently needn't
    for (int idx = 0; idx < seq_size; idx++) {
      _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);

		if((_channel & 8) == 8) {
			_ner_project.ComputeForwardScore(ner_hidden_merge[idx], ner_project[idx]);
		}
		if((_channel & 32) == 32) {
			_sst_project.ComputeForwardScore(sst_hidden_merge[idx], sst_project[idx]);
		}
    }

    int sentBegin = 0;
    for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
  	  int sentLength = sentIdx==0 ? example.sentEnd[sentIdx] : example.sentEnd[sentIdx]-example.sentEnd[sentIdx-1];
  	  int seqIdx = sentBegin;
  	  for(int idx=0;idx<sentLength;idx++, seqIdx++) {
  		  v_recursive_input[sentIdx][idx] += input[seqIdx];
  	  }

        _recursive.ComputeForwardScore(v_recursive_input[sentIdx], example.dep[sentIdx], example.depType[sentIdx],
      		  v_recursive_v_rsc[sentIdx], v_recursive_rsp[sentIdx],
      		  v_recursive_hidden[sentIdx]);

        seqIdx = sentBegin;
  	  for(int idx=0;idx<sentLength;idx++, seqIdx++) {
          vector<Tensor<xpu, 2, dtype> > v_otherInput;
          v_otherInput.push_back(project[seqIdx]);
          v_otherInput.push_back(v_recursive_hidden[sentIdx][idx]);

  		if((_channel & 8) == 8) {
  			  v_otherInput.push_back(ner_project[seqIdx]);
  		}
  		if((_channel & 32) == 32) {
  			  v_otherInput.push_back(sst_project[seqIdx]);
  		}

    		concat(v_otherInput, poolInput[seqIdx]);

  	  }

  	  sentBegin += sentLength;
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


    _represent_transform[0].ComputeForwardScore(beforerepresent, input_span[0]);
    _represent_transform[1].ComputeForwardScore(formerrepresent, input_span[1]);
    _represent_transform[2].ComputeForwardScore(middlerepresent, input_span[2]);
    _represent_transform[3].ComputeForwardScore(latterrepresent, input_span[3]);
    _represent_transform[4].ComputeForwardScore(afterrepresent, input_span[4]);

    _target_attention.ComputeForwardScore(input_span[0], input_span[2], input_span[4],
  		  input_span[1], input_span[3],
          reset_before, reset_middle, reset_after,
			interact);


  concat(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent, poolmerge);
    gatedmerge += interact;

    concat(poolmerge, gatedmerge, allmerge);


    _olayer_linear.ComputeForwardScore(allmerge, output);

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

    for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
  	  FreeSpace(&(v_recursive_input[sentIdx]));
  	  FreeSpace(&(v_recursive_rsp[sentIdx]));
        for(int i=0;i<v_recursive_v_rsc[sentIdx].size();i++)
      	  FreeSpace(&(v_recursive_v_rsc[sentIdx][i]));
  	  FreeSpace(&(v_recursive_hidden[sentIdx]));
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

        FreeSpace(&posprime);
        FreeSpace(&wordWindow);

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

    for (int idm = 0; idm < 5; idm++) {
      FreeSpace(&(input_span[idm]));
    }
    FreeSpace(&reset_before);
    FreeSpace(&reset_middle);
    FreeSpace(&reset_after);
    FreeSpace(&(interact));

    FreeSpace(&beforerepresent);
    FreeSpace(&formerrepresent);
    FreeSpace(&middlerepresent);
	  FreeSpace(&latterrepresent);
	  FreeSpace(&afterrepresent);

    FreeSpace(&poolmerge);
    FreeSpace(&gatedmerge);
    FreeSpace(&allmerge);
    FreeSpace(&output);

    return optLabel;
  }

  dtype computeScore(const Example& example) {
		const vector<Feature>& features = example.m_features;
  int seq_size = features.size();
  int offset = 0;
  int sentNum = example.dep.size();

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

  vector< Tensor<xpu, 3, dtype> > v_recursive_input(sentNum);
  vector< Tensor<xpu, 3, dtype> > v_recursive_rsp(sentNum);
  vector< vector< Tensor<xpu, 3, dtype> > > v_recursive_v_rsc(sentNum);
  vector< Tensor<xpu, 3, dtype> > v_recursive_hidden(sentNum);

  Tensor<xpu, 3, dtype> posprime;
  Tensor<xpu, 3, dtype> wordWindow;
  Tensor<xpu, 3, dtype> sstprime;
  Tensor<xpu, 3, dtype> nerprime;

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
  Tensor<xpu, 2, dtype> gatedmerge;
  Tensor<xpu, 2, dtype> allmerge;
  Tensor<xpu, 2, dtype> output;

  //gated interaction part
  Tensor<xpu, 2, dtype> input_span[5];
  Tensor<xpu, 2, dtype> reset_before, reset_middle, reset_after;
  Tensor<xpu, 2, dtype> interact;

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

  for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
	  int sentLength = sentIdx==0 ? example.sentEnd[sentIdx] : example.sentEnd[sentIdx]-example.sentEnd[sentIdx-1];
	  v_recursive_input[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _inputsize), 0.0);
	  v_recursive_rsp[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _hiddensize), 0.0);
	  v_recursive_hidden[sentIdx] = NewTensor<xpu>(Shape3(sentLength, 1, _hiddensize), 0.0);
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

      posprime = NewTensor<xpu>(Shape3(seq_size, 1, _otherDim), 0.0);
      wordWindow = NewTensor<xpu>(Shape3(seq_size, 1, _wordWindowSize), 0.0);

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

    for (int idm = 0; idm < 5; idm++) {
      input_span[idm] = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    }
    reset_before = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    reset_middle = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    reset_after = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);
    interact = NewTensor<xpu>(Shape2(1, _targetdim), 0.0);


  poolmerge = NewTensor<xpu>(Shape2(1, _poolsize), 0.0);
  gatedmerge = NewTensor<xpu>(Shape2(1, _gatedsize), 0.0);
  allmerge = NewTensor<xpu>(Shape2(1, _poolsize + _gatedsize ), 0.0);
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

		if((_channel & 8) == 8) {
		   _ner.GetEmb(feature.ner, nerprime[idx]);
		}

		   _pos.GetEmb(feature.pos, posprime[idx]);

		if((_channel & 32) == 32) {
		   _sst.GetEmb(feature.sst, sstprime[idx]);
		}

  }

  for (int idx = 0; idx < seq_size; idx++) {
    wordrepresent[idx] += wordprime[idx];
  }

  windowlized(wordrepresent, wordWindow, _wordcontext);
  for (int idx = 0; idx < seq_size; idx++) {
  	concat(wordWindow[idx], posprime[idx], input[idx]);
  }

  _rnn_left.ComputeForwardScore(input, rnn_hidden_left_reset, rnn_hidden_left_afterreset, rnn_hidden_left_update, rnn_hidden_left_current, rnn_hidden_left);
  _rnn_right.ComputeForwardScore(input, rnn_hidden_right_reset, rnn_hidden_right_afterreset, rnn_hidden_right_update, rnn_hidden_right_current, rnn_hidden_right);


	if((_channel & 8) == 8) {
	      _ner_left.ComputeForwardScore(nerprime, ner_hidden_left_reset, ner_hidden_left_afterreset, ner_hidden_left_update, ner_hidden_left_current, ner_hidden_left);
	      _ner_right.ComputeForwardScore(nerprime, ner_hidden_right_reset, ner_hidden_right_afterreset, ner_hidden_right_update, ner_hidden_right_current, ner_hidden_right);
	}
	if((_channel & 32) == 32) {
	      _sst_left.ComputeForwardScore(sstprime, sst_hidden_left_reset, sst_hidden_left_afterreset, sst_hidden_left_update, sst_hidden_left_current, sst_hidden_left);
	      _sst_right.ComputeForwardScore(sstprime, sst_hidden_right_reset, sst_hidden_right_afterreset, sst_hidden_right_update, sst_hidden_right_current, sst_hidden_right);
	}

  for (int idx = 0; idx < seq_size; idx++) {
      concat(rnn_hidden_left[idx], rnn_hidden_right[idx], rnn_hidden_merge[idx]);

		if((_channel & 8) == 8) {
	        concat(ner_hidden_left[idx], ner_hidden_right[idx], ner_hidden_merge[idx]);
		}
		if((_channel & 32) == 32) {
	        concat(sst_hidden_left[idx], sst_hidden_right[idx], sst_hidden_merge[idx]);
		}
  }

  // do we need a convolution? future work, currently needn't
  for (int idx = 0; idx < seq_size; idx++) {
    _tanh_project.ComputeForwardScore(rnn_hidden_merge[idx], project[idx]);

		if((_channel & 8) == 8) {
			_ner_project.ComputeForwardScore(ner_hidden_merge[idx], ner_project[idx]);
		}
		if((_channel & 32) == 32) {
			_sst_project.ComputeForwardScore(sst_hidden_merge[idx], sst_project[idx]);
		}
  }

  int sentBegin = 0;
  for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
	  int sentLength = sentIdx==0 ? example.sentEnd[sentIdx] : example.sentEnd[sentIdx]-example.sentEnd[sentIdx-1];
	  int seqIdx = sentBegin;
	  for(int idx=0;idx<sentLength;idx++, seqIdx++) {
		  v_recursive_input[sentIdx][idx] += input[seqIdx];
	  }

      _recursive.ComputeForwardScore(v_recursive_input[sentIdx], example.dep[sentIdx], example.depType[sentIdx],
    		  v_recursive_v_rsc[sentIdx], v_recursive_rsp[sentIdx],
    		  v_recursive_hidden[sentIdx]);

      seqIdx = sentBegin;
	  for(int idx=0;idx<sentLength;idx++, seqIdx++) {
        vector<Tensor<xpu, 2, dtype> > v_otherInput;
        v_otherInput.push_back(project[seqIdx]);
        v_otherInput.push_back(v_recursive_hidden[sentIdx][idx]);

		if((_channel & 8) == 8) {
			  v_otherInput.push_back(ner_project[seqIdx]);
		}
		if((_channel & 32) == 32) {
			  v_otherInput.push_back(sst_project[seqIdx]);
		}

  		concat(v_otherInput, poolInput[seqIdx]);

	  }

	  sentBegin += sentLength;
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


  _represent_transform[0].ComputeForwardScore(beforerepresent, input_span[0]);
  _represent_transform[1].ComputeForwardScore(formerrepresent, input_span[1]);
  _represent_transform[2].ComputeForwardScore(middlerepresent, input_span[2]);
  _represent_transform[3].ComputeForwardScore(latterrepresent, input_span[3]);
  _represent_transform[4].ComputeForwardScore(afterrepresent, input_span[4]);

  _target_attention.ComputeForwardScore(input_span[0], input_span[2], input_span[4],
		  input_span[1], input_span[3],
        reset_before, reset_middle, reset_after,
			interact);


concat(beforerepresent, formerrepresent, middlerepresent, latterrepresent, afterrepresent, poolmerge);
  gatedmerge += interact;

  concat(poolmerge, gatedmerge, allmerge);


  _olayer_linear.ComputeForwardScore(allmerge, output);

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

  for(int sentIdx=0;sentIdx<sentNum;sentIdx++) {
	  FreeSpace(&(v_recursive_input[sentIdx]));
	  FreeSpace(&(v_recursive_rsp[sentIdx]));
      for(int i=0;i<v_recursive_v_rsc[sentIdx].size();i++)
    	  FreeSpace(&(v_recursive_v_rsc[sentIdx][i]));
	  FreeSpace(&(v_recursive_hidden[sentIdx]));
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

      FreeSpace(&posprime);
      FreeSpace(&wordWindow);

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

  for (int idm = 0; idm < 5; idm++) {
    FreeSpace(&(input_span[idm]));
  }
  FreeSpace(&reset_before);
  FreeSpace(&reset_middle);
  FreeSpace(&reset_after);
  FreeSpace(&(interact));

  FreeSpace(&beforerepresent);
  FreeSpace(&formerrepresent);
  FreeSpace(&middlerepresent);
	  FreeSpace(&latterrepresent);
	  FreeSpace(&afterrepresent);

  FreeSpace(&poolmerge);
  FreeSpace(&gatedmerge);
  FreeSpace(&allmerge);
  FreeSpace(&output);

  return cost;
}

  void updateParams(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
    _tanh_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _olayer_linear.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _rnn_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _rnn_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _recursive.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    for (int idx = 0; idx < 5; idx++) {
      _represent_transform[idx].updateAdaGrad(nnRegular, adaAlpha, adaEps);
    }
    _target_attention.updateAdaGrad(nnRegular, adaAlpha, adaEps);

    _words.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _pos.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _ner.updateAdaGrad(nnRegular, adaAlpha, adaEps);
    _sst.updateAdaGrad(nnRegular, adaAlpha, adaEps);

	if((_channel & 8) == 8) {
	    _ner_left.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _ner_right.updateAdaGrad(nnRegular, adaAlpha, adaEps);
	    _ner_project.updateAdaGrad(nnRegular, adaAlpha, adaEps);
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

	    checkgrad(examples, _recursive._recursive_p._W, _recursive._recursive_p._gradW, "_recursive._recursive_p._W", iter);
	    checkgrad(examples, _recursive._recursive_r_other._W, _recursive._recursive_r_other._gradW, "_recursive._recursive_r_other._W", iter);
	    for(int i=0;i<_recursive._recursive_r.size();i++) {
	    	stringstream ss;
	    	ss<<"_recursive._recursive_r["<<i<<"]._W";
	    	checkgrad(examples, _recursive._recursive_r[i]._W, _recursive._recursive_r[i]._gradW, ss.str(), iter);
	    }
	    checkgrad(examples, _recursive._b, _recursive._gradb, "_recursive._b", iter);

	    checkgrad(examples, _target_attention._reset1._W1, _target_attention._reset1._gradW1, "_target_attention._reset1._W1", iter);
	    checkgrad(examples, _target_attention._reset1._W2, _target_attention._reset1._gradW2, "_target_attention._reset1._W2", iter);
	    checkgrad(examples, _target_attention._reset1._W3, _target_attention._reset1._gradW3, "_target_attention._reset1._W3", iter);
	    checkgrad(examples, _target_attention._reset1._b, _target_attention._reset1._gradb, "_target_attention._reset1._b", iter);

	    checkgrad(examples, _target_attention._reset2._W1, _target_attention._reset2._gradW1, "_target_attention._reset2._W1", iter);
	    checkgrad(examples, _target_attention._reset2._W2, _target_attention._reset2._gradW2, "_target_attention._reset2._W2", iter);
	    checkgrad(examples, _target_attention._reset2._W3, _target_attention._reset2._gradW3, "_target_attention._reset2._W3", iter);
	    checkgrad(examples, _target_attention._reset2._b, _target_attention._reset2._gradb, "_target_attention._reset2._b", iter);

	    checkgrad(examples, _target_attention._reset3._W1, _target_attention._reset3._gradW1, "_target_attention._reset3._W1", iter);
	    checkgrad(examples, _target_attention._reset3._W2, _target_attention._reset3._gradW2, "_target_attention._reset3._W2", iter);
	    checkgrad(examples, _target_attention._reset3._W3, _target_attention._reset3._gradW3, "_target_attention._reset3._W3", iter);
	    checkgrad(examples, _target_attention._reset3._b, _target_attention._reset3._gradb, "_target_attention._reset3._b", iter);

	    checkgrad(examples, _target_attention._recursive_tilde._W1, _target_attention._recursive_tilde._gradW1, "_target_attention._recursive_tilde._W1", iter);
	    checkgrad(examples, _target_attention._recursive_tilde._W2, _target_attention._recursive_tilde._gradW2, "_target_attention._recursive_tilde._W2", iter);
	    checkgrad(examples, _target_attention._recursive_tilde._W3, _target_attention._recursive_tilde._gradW3, "_target_attention._recursive_tilde._W3", iter);
	    checkgrad(examples, _target_attention._recursive_tilde._b, _target_attention._recursive_tilde._gradb, "_target_attention._recursive_tilde._b", iter);

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

	    for (int idx = 0; idx < 5; idx++) {
	      stringstream ssposition;
	      ssposition << "[" << idx << "]";

	      checkgrad(examples, _represent_transform[idx]._W, _represent_transform[idx]._gradW, "_represent_transform" + ssposition.str() + "._W", iter);
	      checkgrad(examples, _represent_transform[idx]._b, _represent_transform[idx]._gradb, "_represent_transform" + ssposition.str() + "._b", iter);
	    }

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
