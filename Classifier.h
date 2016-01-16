/*
 * Classifier.h
 *
 *  Created on: Dec 28, 2015
 *      Author: fox
 */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <iostream>

#include <assert.h>
#include "Example.h"
#include "Metric.h"
#include "N3L.h"
#include "Options.h"
#include "QuinLayer.h"
#include "Attention.h"
#include "EightLayer.h"
#include "AttentionPooling.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class Classifier {
public:
	Options options;
	LookupTable<xpu> _words;
	int _wordSize;
	int _wordDim;
	int _outputSize;

	// input
	LSTM<xpu> unit_before;
	LSTM<xpu> unit_entityFormer;
	LSTM<xpu> unit_entityLatter;
	LSTM<xpu> unit_middle;
	LSTM<xpu> unit_after;
/*	Attention<xpu> unit_att_before;
	Attention<xpu> unit_att_middle;
	Attention<xpu> unit_att_after;*/
	AttentionPooling<xpu> unit_att_before;
	AttentionPooling<xpu> unit_att_middle;
	AttentionPooling<xpu> unit_att_after;
	// hidden
	QuinLayer<xpu> hidden_layer;
	EightLayer<xpu> combine_hidden_layer; // if input_represent is 2
	// output
	UniLayer<xpu> output_layer;
	// softmax loss corresponds no class in n3l

	Metric _eval;

	Classifier(const Options& options):options(options) {

	}
	virtual ~Classifier() {

	}

	void release() {

		output_layer.release();

		if(options.input_represent == 2)
			combine_hidden_layer.release();
		else
			hidden_layer.release();

		unit_att_before.release();
		unit_att_middle.release();
		unit_att_after.release();

		unit_before.release();
		unit_entityFormer.release();
		unit_entityLatter.release();
		unit_middle.release();
		unit_after.release();

		_words.release();
	}


	void init(const NRMat<dtype>& wordEmb) {

	    _wordSize = wordEmb.nrows();
	    _wordDim = wordEmb.ncols();
	    _outputSize = 2; // has relation or not

	    _words.initial(wordEmb);
	    _words.setEmbFineTune(options.wordEmbFineTune);

	    unit_before.initial(options.context_embsize, _wordDim, (int)time(0));
	    unit_entityFormer.initial(options.entity_embsize, _wordDim, (int)time(0));
	    unit_entityLatter.initial(options.entity_embsize, _wordDim, (int)time(0));
	    unit_middle.initial(options.context_embsize, _wordDim, (int)time(0));
	    unit_after.initial(options.context_embsize, _wordDim, (int)time(0));

/*	    unit_att_before.initial(options.wordEmbSize, options.entity_embsize, options.context_embsize);
	    unit_att_middle.initial(options.wordEmbSize, options.entity_embsize, options.context_embsize);
	    unit_att_after.initial(options.wordEmbSize, options.entity_embsize, options.context_embsize);*/
	    unit_att_before.initial(options.wordEmbSize, options.context_embsize, true, (int)time(0));
	    unit_att_middle.initial(options.wordEmbSize, options.context_embsize, true, (int)time(0));
	    unit_att_after.initial(options.wordEmbSize, options.context_embsize, true, (int)time(0));

	    if(options.input_represent == 2) {
	    	combine_hidden_layer.initial(options.hiddenSize, options.context_embsize, options.wordEmbSize,
	    			options.entity_embsize, options.context_embsize, options.wordEmbSize,
					options.entity_embsize, options.context_embsize, options.wordEmbSize,
					true, (int)time(0), 0);
	    } else if (options.input_represent == 1) {
	    	hidden_layer.initial(options.hiddenSize, options.wordEmbSize,
					options.entity_embsize, options.wordEmbSize, options.entity_embsize,
					options.wordEmbSize,
					true, (int)time(0), 0);
	    } else {
	    	hidden_layer.initial(options.hiddenSize, options.context_embsize,
	    		options.entity_embsize, options.context_embsize, options.entity_embsize,
	    		options.context_embsize,
	    		true, (int)time(0), 0);
	    }



	    output_layer.initial(_outputSize, options.hiddenSize, true, (int)time(0), 2);

	}

	void predict(const Example& example, vector<double>& scores) {
		int beforeSize = example.m_before.size();
		int enFormerSize = example.m_entityFormer.size();
		int enLatterSize = example.m_entityLatter.size();
		int middleSize = example.m_middle.size();
		int afterSize = example.m_after.size();

		vector<Tensor<xpu, 2, dtype> > input_before(beforeSize);
		for (int idx = 0; idx < beforeSize; idx++) {
			input_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
		}

		vector<Tensor<xpu, 2, dtype> > input_entityFormer(enFormerSize);
		for (int idx = 0; idx < enFormerSize; idx++) {
			input_entityFormer[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
		}

		vector<Tensor<xpu, 2, dtype> > input_entityLatter(enLatterSize);
		for (int idx = 0; idx < enLatterSize; idx++) {
			input_entityLatter[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
		}

		vector<Tensor<xpu, 2, dtype> > input_middle(middleSize);
		for (int idx = 0; idx < middleSize; idx++) {
			input_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
		}

		vector<Tensor<xpu, 2, dtype> > input_after(afterSize);
		for (int idx = 0; idx < afterSize; idx++) {
			input_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
		}

		Tensor<xpu, 2, dtype> hidden = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);


		Tensor<xpu, 2, dtype> output = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);


		//forward propagation
		for (int idx = 0; idx < beforeSize; idx++) {
			_words.GetEmb(example.m_before[idx], input_before[idx]);
		}

		for (int idx = 0; idx < enFormerSize; idx++) {
			_words.GetEmb(example.m_entityFormer[idx], input_entityFormer[idx]);
		}
		for (int idx = 0; idx < enLatterSize; idx++) {
			_words.GetEmb(example.m_entityLatter[idx], input_entityLatter[idx]);
		}
		for (int idx = 0; idx < middleSize; idx++) {
			_words.GetEmb(example.m_middle[idx], input_middle[idx]);
		}
		for (int idx = 0; idx < afterSize; idx++) {
			_words.GetEmb(example.m_after[idx], input_after[idx]);
		}

		// compute before unit
		vector<Tensor<xpu, 2, dtype> > iy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > oy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > fy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > mcy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > cy_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > my_before(beforeSize);
		vector<Tensor<xpu, 2, dtype> > y_before(beforeSize);
		for (int idx = 0; idx < beforeSize; idx++) {
			iy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			oy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			fy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			mcy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			cy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			my_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			y_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
		}
		unit_before.ComputeForwardScore(input_before, iy_before, oy_before,
				fy_before, mcy_before,cy_before, my_before, y_before);

		// compute entity former unit
		vector<Tensor<xpu, 2, dtype> > iy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > my_entityFormer(enFormerSize);
		vector<Tensor<xpu, 2, dtype> > y_entityFormer(enFormerSize);
		for (int idx = 0; idx < enFormerSize; idx++) {
			iy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			oy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			fy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			mcy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			cy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			my_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			y_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
		}
		unit_entityFormer.ComputeForwardScore(input_entityFormer, iy_entityFormer, oy_entityFormer,
				fy_entityFormer, mcy_entityFormer,cy_entityFormer, my_entityFormer, y_entityFormer);

		// compute entity latter unit
		vector<Tensor<xpu, 2, dtype> > iy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > oy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > fy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > cy_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > my_entityLatter(enLatterSize);
		vector<Tensor<xpu, 2, dtype> > y_entityLatter(enLatterSize);
		for (int idx = 0; idx < enLatterSize; idx++) {
			iy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			oy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			fy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			mcy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			cy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			my_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			y_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
		}
		unit_entityLatter.ComputeForwardScore(input_entityLatter, iy_entityLatter, oy_entityLatter,
				fy_entityLatter, mcy_entityLatter,cy_entityLatter, my_entityLatter, y_entityLatter);

		// compute middle unit
		vector<Tensor<xpu, 2, dtype> > iy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > oy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > fy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > mcy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > cy_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > my_middle(middleSize);
		vector<Tensor<xpu, 2, dtype> > y_middle(middleSize);
		for (int idx = 0; idx < middleSize; idx++) {
			iy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			oy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			fy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			mcy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			cy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			my_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			y_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
		}
		unit_middle.ComputeForwardScore(input_middle, iy_middle, oy_middle,
				fy_middle, mcy_middle,cy_middle, my_middle, y_middle);

		// compute after unit
		vector<Tensor<xpu, 2, dtype> > iy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > oy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > fy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > mcy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > cy_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > my_after(afterSize);
		vector<Tensor<xpu, 2, dtype> > y_after(afterSize);
		for (int idx = 0; idx < afterSize; idx++) {
			iy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			oy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			fy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			mcy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			cy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			my_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
			y_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
		}
		unit_after.ComputeForwardScore(input_after, iy_after, oy_after,
				fy_after, mcy_after,cy_after, my_after, y_after);


		if(options.input_represent ==1) {
			// attention before
			vector<Tensor<xpu, 2, dtype> > xMExp_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xExp_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_before(beforeSize);
			for (int idx = 0; idx < beforeSize; idx++) {
				xMExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xPoolIndex_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}
			Tensor<xpu, 2, dtype> xSum_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			Tensor<xpu, 2, dtype> y_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*			unit_att_before.ComputeForwardScore(input_before, y_entityFormer[enFormerSize-1],
					y_entityLatter[enLatterSize-1], y_middle[middleSize-1], y_after[afterSize-1],
					xMExp_before, xExp_before, xSum_before,
					xPoolIndex_before, y_att_before);*/
			unit_att_before.ComputeForwardScore(input_before, input_before,
					xMExp_before, xExp_before, xSum_before,
					xPoolIndex_before, y_att_before);

			// attention middle
			vector<Tensor<xpu, 2, dtype> > xMExp_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > xExp_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle(middleSize);
			for (int idx = 0; idx < middleSize; idx++) {
				xMExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xPoolIndex_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}
			Tensor<xpu, 2, dtype> xSum_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			Tensor<xpu, 2, dtype> y_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*			unit_att_middle.ComputeForwardScore(input_middle, y_entityFormer[enFormerSize-1],
					y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_after[afterSize-1],
					xMExp_middle, xExp_middle, xSum_middle,
					xPoolIndex_middle, y_att_middle);*/
			unit_att_middle.ComputeForwardScore(input_middle, input_middle,
					xMExp_middle, xExp_middle, xSum_middle,
					xPoolIndex_middle, y_att_middle);

			// attention after
			vector<Tensor<xpu, 2, dtype> > xMExp_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > xExp_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_after(afterSize);
			for (int idx = 0; idx < afterSize; idx++) {
				xMExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xPoolIndex_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}
			Tensor<xpu, 2, dtype> xSum_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			Tensor<xpu, 2, dtype> y_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*			unit_att_after.ComputeForwardScore(input_after, y_entityFormer[enFormerSize-1],
					y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_middle[middleSize-1],
					xMExp_after, xExp_after, xSum_after,
					xPoolIndex_after, y_att_after);*/
			unit_att_after.ComputeForwardScore(input_after, input_after,
					xMExp_after, xExp_after, xSum_after,
					xPoolIndex_after, y_att_after);

			// input -> hidden
			hidden_layer.ComputeForwardScore(y_att_before,
					y_entityFormer[enFormerSize-1], y_att_middle, y_entityLatter[enLatterSize-1],
					y_att_after, hidden);

			// hidden -> output
			output_layer.ComputeForwardScore(hidden, output);

			scores[0] = output[0][0];
			scores[1] = output[0][1];

			// release
			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(xMExp_before[idx]));
				FreeSpace(&(xExp_before[idx]));
				FreeSpace(&(xPoolIndex_before[idx]));
			}
			FreeSpace(&(xSum_before));
			FreeSpace(&(y_att_before));

			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(xMExp_middle[idx]));
				FreeSpace(&(xExp_middle[idx]));
				FreeSpace(&(xPoolIndex_middle[idx]));
			}
			FreeSpace(&(xSum_middle));
			FreeSpace(&(y_att_middle));

			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(xMExp_after[idx]));
				FreeSpace(&(xExp_after[idx]));
				FreeSpace(&(xPoolIndex_after[idx]));
			}
			FreeSpace(&(xSum_after));
			FreeSpace(&(y_att_after));

		} else if(options.input_represent == 2) {
			// attention before
			vector<Tensor<xpu, 2, dtype> > xMExp_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xExp_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_before(beforeSize);
			for (int idx = 0; idx < beforeSize; idx++) {
				xMExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xPoolIndex_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}
			Tensor<xpu, 2, dtype> xSum_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			Tensor<xpu, 2, dtype> y_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*			unit_att_before.ComputeForwardScore(input_before, y_entityFormer[enFormerSize-1],
					y_entityLatter[enLatterSize-1], y_middle[middleSize-1], y_after[afterSize-1],
					xMExp_before, xExp_before, xSum_before,
					xPoolIndex_before, y_att_before);*/
			unit_att_before.ComputeForwardScore(input_before, input_before,
					xMExp_before, xExp_before, xSum_before,
					xPoolIndex_before, y_att_before);

			// attention middle
			vector<Tensor<xpu, 2, dtype> > xMExp_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > xExp_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle(middleSize);
			for (int idx = 0; idx < middleSize; idx++) {
				xMExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xPoolIndex_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}
			Tensor<xpu, 2, dtype> xSum_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			Tensor<xpu, 2, dtype> y_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*			unit_att_middle.ComputeForwardScore(input_middle, y_entityFormer[enFormerSize-1],
					y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_after[afterSize-1],
					xMExp_middle, xExp_middle, xSum_middle,
					xPoolIndex_middle, y_att_middle);*/
			unit_att_middle.ComputeForwardScore(input_middle, input_middle,
					xMExp_middle, xExp_middle, xSum_middle,
					xPoolIndex_middle, y_att_middle);

			// attention after
			vector<Tensor<xpu, 2, dtype> > xMExp_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > xExp_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > xPoolIndex_after(afterSize);
			for (int idx = 0; idx < afterSize; idx++) {
				xMExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				xPoolIndex_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}
			Tensor<xpu, 2, dtype> xSum_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			Tensor<xpu, 2, dtype> y_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*			unit_att_after.ComputeForwardScore(input_after, y_entityFormer[enFormerSize-1],
					y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_middle[middleSize-1],
					xMExp_after, xExp_after, xSum_after,
					xPoolIndex_after, y_att_after);*/
			unit_att_after.ComputeForwardScore(input_after, input_after,
					xMExp_after, xExp_after, xSum_after,
					xPoolIndex_after, y_att_after);

			// input -> hidden
			combine_hidden_layer.ComputeForwardScore(y_before[beforeSize-1], y_att_before,
					y_entityFormer[enFormerSize-1], y_middle[middleSize-1], y_att_middle,
					y_entityLatter[enLatterSize-1], y_after[afterSize-1], y_att_after,
					hidden);

			// hidden -> output
			output_layer.ComputeForwardScore(hidden, output);

			scores[0] = output[0][0];
			scores[1] = output[0][1];

			// release
			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(xMExp_before[idx]));
				FreeSpace(&(xExp_before[idx]));
				FreeSpace(&(xPoolIndex_before[idx]));
			}
			FreeSpace(&(xSum_before));
			FreeSpace(&(y_att_before));

			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(xMExp_middle[idx]));
				FreeSpace(&(xExp_middle[idx]));
				FreeSpace(&(xPoolIndex_middle[idx]));
			}
			FreeSpace(&(xSum_middle));
			FreeSpace(&(y_att_middle));

			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(xMExp_after[idx]));
				FreeSpace(&(xExp_after[idx]));
				FreeSpace(&(xPoolIndex_after[idx]));
			}
			FreeSpace(&(xSum_after));
			FreeSpace(&(y_att_after));
		} else {
			// input -> hidden
			hidden_layer.ComputeForwardScore(y_before[beforeSize-1],
				y_entityFormer[enFormerSize-1], y_middle[middleSize-1], y_entityLatter[enLatterSize-1],
				y_after[afterSize-1], hidden);

			// hidden -> output
			output_layer.ComputeForwardScore(hidden, output);

			scores[0] = output[0][0];
			scores[1] = output[0][1];
		}



		// release all the stuff
		for (int idx = 0; idx < beforeSize; idx++) {
			FreeSpace(&(input_before[idx]));
		}

		for (int idx = 0; idx < enFormerSize; idx++) {
			FreeSpace(&(input_entityFormer[idx]));
		}

		for (int idx = 0; idx < enLatterSize; idx++) {
			FreeSpace(&(input_entityLatter[idx]));
		}

		for (int idx = 0; idx < middleSize; idx++) {
			FreeSpace(&(input_middle[idx]));
		}

		for (int idx = 0; idx < afterSize; idx++) {
			FreeSpace(&(input_after[idx]));
		}

		FreeSpace(&hidden);

		FreeSpace(&output);

		for (int idx = 0; idx < beforeSize; idx++) {
			FreeSpace(&(iy_before[idx]));
			FreeSpace(&(oy_before[idx]));
			FreeSpace(&(fy_before[idx]));
			FreeSpace(&(mcy_before[idx]));
			FreeSpace(&(cy_before[idx]));
			FreeSpace(&(my_before[idx]));
			FreeSpace(&(y_before[idx]));
		}

		for (int idx = 0; idx < enFormerSize; idx++) {
			FreeSpace(&(iy_entityFormer[idx]));
			FreeSpace(&(oy_entityFormer[idx]));
			FreeSpace(&(fy_entityFormer[idx]));
			FreeSpace(&(mcy_entityFormer[idx]));
			FreeSpace(&(cy_entityFormer[idx]));
			FreeSpace(&(my_entityFormer[idx]));
			FreeSpace(&(y_entityFormer[idx]));
		}

		for (int idx = 0; idx < enLatterSize; idx++) {
			FreeSpace(&(iy_entityLatter[idx]));
			FreeSpace(&(oy_entityLatter[idx]));
			FreeSpace(&(fy_entityLatter[idx]));
			FreeSpace(&(mcy_entityLatter[idx]));
			FreeSpace(&(cy_entityLatter[idx]));
			FreeSpace(&(my_entityLatter[idx]));
			FreeSpace(&(y_entityLatter[idx]));
		}

		for (int idx = 0; idx < middleSize; idx++) {
			FreeSpace(&(iy_middle[idx]));
			FreeSpace(&(oy_middle[idx]));
			FreeSpace(&(fy_middle[idx]));
			FreeSpace(&(mcy_middle[idx]));
			FreeSpace(&(cy_middle[idx]));
			FreeSpace(&(my_middle[idx]));
			FreeSpace(&(y_middle[idx]));
		}

		for (int idx = 0; idx < afterSize; idx++) {
			FreeSpace(&(iy_after[idx]));
			FreeSpace(&(oy_after[idx]));
			FreeSpace(&(fy_after[idx]));
			FreeSpace(&(mcy_after[idx]));
			FreeSpace(&(cy_after[idx]));
			FreeSpace(&(my_after[idx]));
			FreeSpace(&(y_after[idx]));
		}

	}

	dtype process(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;
		int offset = 0;
		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];
			int enFormerSize = example.m_entityFormer.size();
			int enLatterSize = example.m_entityLatter.size();
			int middleSize = example.m_middle.size();
			int beforeSize = example.m_before.size();
			int afterSize = example.m_after.size();

			vector<Tensor<xpu, 2, dtype> > input_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mask_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_before(beforeSize);
			for (int idx = 0; idx < beforeSize; idx++) {
				input_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				mask_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				inputLoss_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}

			vector<Tensor<xpu, 2, dtype> > input_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityFormer(enFormerSize);
			for (int idx = 0; idx < enFormerSize; idx++) {
				input_entityFormer[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				mask_entityFormer[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				inputLoss_entityFormer[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}

			vector<Tensor<xpu, 2, dtype> > input_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mask_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_entityLatter(enLatterSize);
			for (int idx = 0; idx < enLatterSize; idx++) {
				input_entityLatter[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				mask_entityLatter[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				inputLoss_entityLatter[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}

			vector<Tensor<xpu, 2, dtype> > input_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > mask_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_middle(middleSize);
			for (int idx = 0; idx < middleSize; idx++) {
				input_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				mask_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				inputLoss_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}

			vector<Tensor<xpu, 2, dtype> > input_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > mask_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > inputLoss_after(afterSize);
			for (int idx = 0; idx < afterSize; idx++) {
				input_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				mask_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				inputLoss_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
			}

			Tensor<xpu, 2, dtype> hidden = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);
			Tensor<xpu, 2, dtype> hiddenLoss = NewTensor<xpu>(Shape2(1, options.hiddenSize), d_zero);


			Tensor<xpu, 2, dtype> output = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);
			Tensor<xpu, 2, dtype> outputLoss = NewTensor<xpu>(Shape2(1, _outputSize), d_zero);


			//forward propagation
			for (int idx = 0; idx < beforeSize; idx++) {
				srand(iter * example_num + count + idx);
				_words.GetEmb(example.m_before[idx], input_before[idx]);
				dropoutcol(mask_before[idx], options.dropProb);
				input_before[idx] = input_before[idx] * mask_before[idx];
			}
			for (int idx = 0; idx < enFormerSize; idx++) {
				srand(iter * example_num + count + idx);
				_words.GetEmb(example.m_entityFormer[idx], input_entityFormer[idx]);
				dropoutcol(mask_entityFormer[idx], options.dropProb);
				input_entityFormer[idx] = input_entityFormer[idx] * mask_entityFormer[idx];
			}
			for (int idx = 0; idx < enLatterSize; idx++) {
				srand(iter * example_num + count + idx);
				_words.GetEmb(example.m_entityLatter[idx], input_entityLatter[idx]);
				dropoutcol(mask_entityLatter[idx], options.dropProb);
				input_entityLatter[idx] = input_entityLatter[idx] * mask_entityLatter[idx];
			}
			for (int idx = 0; idx < middleSize; idx++) {
				srand(iter * example_num + count + idx);
				_words.GetEmb(example.m_middle[idx], input_middle[idx]);
				dropoutcol(mask_middle[idx], options.dropProb);
				input_middle[idx] = input_middle[idx] * mask_middle[idx];
			}
			for (int idx = 0; idx < afterSize; idx++) {
				srand(iter * example_num + count + idx);
				_words.GetEmb(example.m_after[idx], input_after[idx]);
				dropoutcol(mask_after[idx], options.dropProb);
				input_after[idx] = input_after[idx] * mask_after[idx];
			}

			// compute before unit
			vector<Tensor<xpu, 2, dtype> > iy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > oy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > fy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > mcy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > cy_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > my_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > y_before(beforeSize);
			vector<Tensor<xpu, 2, dtype> > loss_before(beforeSize);
			for (int idx = 0; idx < beforeSize; idx++) {
				iy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				loss_before[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);;
			}
			unit_before.ComputeForwardScore(input_before, iy_before, oy_before,
					fy_before, mcy_before,cy_before, my_before, y_before);

			// compute entity former unit
			vector<Tensor<xpu, 2, dtype> > iy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > my_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > y_entityFormer(enFormerSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityFormer(enFormerSize);
			for (int idx = 0; idx < enFormerSize; idx++) {
				iy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				loss_entityFormer[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
			}
			unit_entityFormer.ComputeForwardScore(input_entityFormer, iy_entityFormer, oy_entityFormer,
					fy_entityFormer, mcy_entityFormer,cy_entityFormer, my_entityFormer, y_entityFormer);

			// compute entity latter unit
			vector<Tensor<xpu, 2, dtype> > iy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > oy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > fy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > cy_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > my_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > y_entityLatter(enLatterSize);
			vector<Tensor<xpu, 2, dtype> > loss_entityLatter(enLatterSize);
			for (int idx = 0; idx < enLatterSize; idx++) {
				iy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				oy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				fy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				mcy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				cy_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				my_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				y_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);
				loss_entityLatter[idx] = NewTensor<xpu>(Shape2(1, options.entity_embsize), d_zero);;
			}
			unit_entityLatter.ComputeForwardScore(input_entityLatter, iy_entityLatter, oy_entityLatter,
					fy_entityLatter, mcy_entityLatter,cy_entityLatter, my_entityLatter, y_entityLatter);

			// compute middle unit
			vector<Tensor<xpu, 2, dtype> > iy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > oy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > fy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > mcy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > cy_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > my_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > y_middle(middleSize);
			vector<Tensor<xpu, 2, dtype> > loss_middle(middleSize);
			for (int idx = 0; idx < middleSize; idx++) {
				iy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				loss_middle[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);;
			}
			unit_middle.ComputeForwardScore(input_middle, iy_middle, oy_middle,
					fy_middle, mcy_middle,cy_middle, my_middle, y_middle);

			// compute after unit
			vector<Tensor<xpu, 2, dtype> > iy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > oy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > fy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > mcy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > cy_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > my_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > y_after(afterSize);
			vector<Tensor<xpu, 2, dtype> > loss_after(afterSize);
			for (int idx = 0; idx < afterSize; idx++) {
				iy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				oy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				fy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				mcy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				cy_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				my_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				y_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);
				loss_after[idx] = NewTensor<xpu>(Shape2(1, options.context_embsize), d_zero);;
			}
			unit_after.ComputeForwardScore(input_after, iy_after, oy_after,
					fy_after, mcy_after,cy_after, my_after, y_after);


			if(options.input_represent ==1) {
				// attention before
				vector<Tensor<xpu, 2, dtype> > xMExp_before(beforeSize);
				vector<Tensor<xpu, 2, dtype> > xExp_before(beforeSize);
				vector<Tensor<xpu, 2, dtype> > xPoolIndex_before(beforeSize);
				for (int idx = 0; idx < beforeSize; idx++) {
					xMExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				Tensor<xpu, 2, dtype> xSum_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> y_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> ly_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*				unit_att_before.ComputeForwardScore(input_before, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_middle[middleSize-1], y_after[afterSize-1],
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before);*/
				unit_att_before.ComputeForwardScore(input_before, input_before,
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before);

				// attention middle
				vector<Tensor<xpu, 2, dtype> > xMExp_middle(middleSize);
				vector<Tensor<xpu, 2, dtype> > xExp_middle(middleSize);
				vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle(middleSize);
				for (int idx = 0; idx < middleSize; idx++) {
					xMExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				Tensor<xpu, 2, dtype> xSum_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> y_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> ly_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*				unit_att_middle.ComputeForwardScore(input_middle, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_after[afterSize-1],
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle);*/
				unit_att_middle.ComputeForwardScore(input_middle, input_middle,
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle);

				// attention after
				vector<Tensor<xpu, 2, dtype> > xMExp_after(afterSize);
				vector<Tensor<xpu, 2, dtype> > xExp_after(afterSize);
				vector<Tensor<xpu, 2, dtype> > xPoolIndex_after(afterSize);
				for (int idx = 0; idx < afterSize; idx++) {
					xMExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				Tensor<xpu, 2, dtype> xSum_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> y_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> ly_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*				unit_att_after.ComputeForwardScore(input_after, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_middle[middleSize-1],
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after);*/
				unit_att_after.ComputeForwardScore(input_after, input_after,
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after);

				// input -> hidden
				hidden_layer.ComputeForwardScore(y_att_before,
						y_entityFormer[enFormerSize-1], y_att_middle, y_entityLatter[enLatterSize-1],
						y_att_after, hidden);

				// hidden -> output
				output_layer.ComputeForwardScore(hidden, output);

				// get delta for each output
				cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

				// loss backward propagation
				// output
				output_layer.ComputeBackwardLoss(hidden, output, outputLoss, hiddenLoss);

				// hidden
				hidden_layer.ComputeBackwardLoss(y_att_before, y_entityFormer[enFormerSize-1],
						y_att_middle, y_entityLatter[enLatterSize-1], y_att_after,
						hidden, hiddenLoss,
						ly_att_before, loss_entityFormer[enFormerSize-1], ly_att_middle,
						loss_entityLatter[enLatterSize-1], ly_att_after);

				// attention
/*				unit_att_before.ComputeBackwardLoss(input_before, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_middle[middleSize-1], y_after[afterSize-1],
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before,
				      ly_att_before, inputLoss_before,
					  loss_entityFormer[enFormerSize-1], loss_entityLatter[enLatterSize-1],
					  loss_middle[middleSize-1], loss_after[afterSize-1]);

				unit_att_middle.ComputeBackwardLoss(input_middle, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_after[afterSize-1],
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle,
						ly_att_middle, inputLoss_middle,
						loss_entityFormer[enFormerSize-1], loss_entityLatter[enLatterSize-1],
						loss_before[beforeSize-1], loss_after[afterSize-1]);

				unit_att_after.ComputeBackwardLoss(input_after, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_middle[middleSize-1],
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after,
						ly_att_after, inputLoss_after,
						loss_entityFormer[enFormerSize-1], loss_entityLatter[enLatterSize-1],
						loss_before[beforeSize-1], loss_middle[middleSize-1]);*/

				unit_att_before.ComputeBackwardLoss(input_before, input_before,
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before,
						ly_att_before, inputLoss_before, inputLoss_before);

				unit_att_middle.ComputeBackwardLoss(input_middle, input_middle,
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle,
						ly_att_middle, inputLoss_middle, inputLoss_middle);

				unit_att_after.ComputeBackwardLoss(input_after, input_after,
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after,
						ly_att_after, inputLoss_after, inputLoss_after);


				// release
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(xMExp_before[idx]));
					FreeSpace(&(xExp_before[idx]));
					FreeSpace(&(xPoolIndex_before[idx]));
				}
				FreeSpace(&(xSum_before));
				FreeSpace(&(y_att_before));
				FreeSpace(&(ly_att_before));

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(xMExp_middle[idx]));
					FreeSpace(&(xExp_middle[idx]));
					FreeSpace(&(xPoolIndex_middle[idx]));
				}
				FreeSpace(&(xSum_middle));
				FreeSpace(&(y_att_middle));
				FreeSpace(&(ly_att_middle));

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(xMExp_after[idx]));
					FreeSpace(&(xExp_after[idx]));
					FreeSpace(&(xPoolIndex_after[idx]));
				}
				FreeSpace(&(xSum_after));
				FreeSpace(&(y_att_after));
				FreeSpace(&(ly_att_after));

			} else if(options.input_represent == 2) {
				// attention before
				vector<Tensor<xpu, 2, dtype> > xMExp_before(beforeSize);
				vector<Tensor<xpu, 2, dtype> > xExp_before(beforeSize);
				vector<Tensor<xpu, 2, dtype> > xPoolIndex_before(beforeSize);
				for (int idx = 0; idx < beforeSize; idx++) {
					xMExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_before[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				Tensor<xpu, 2, dtype> xSum_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> y_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> ly_att_before = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*				unit_att_before.ComputeForwardScore(input_before, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_middle[middleSize-1], y_after[afterSize-1],
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before);*/
				unit_att_before.ComputeForwardScore(input_before, input_before,
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before);

				// attention middle
				vector<Tensor<xpu, 2, dtype> > xMExp_middle(middleSize);
				vector<Tensor<xpu, 2, dtype> > xExp_middle(middleSize);
				vector<Tensor<xpu, 2, dtype> > xPoolIndex_middle(middleSize);
				for (int idx = 0; idx < middleSize; idx++) {
					xMExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_middle[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				Tensor<xpu, 2, dtype> xSum_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> y_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> ly_att_middle = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*				unit_att_middle.ComputeForwardScore(input_middle, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_after[afterSize-1],
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle);*/
				unit_att_middle.ComputeForwardScore(input_middle, input_middle,
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle);

				// attention after
				vector<Tensor<xpu, 2, dtype> > xMExp_after(afterSize);
				vector<Tensor<xpu, 2, dtype> > xExp_after(afterSize);
				vector<Tensor<xpu, 2, dtype> > xPoolIndex_after(afterSize);
				for (int idx = 0; idx < afterSize; idx++) {
					xMExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xExp_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
					xPoolIndex_after[idx] = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				}
				Tensor<xpu, 2, dtype> xSum_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> y_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);
				Tensor<xpu, 2, dtype> ly_att_after = NewTensor<xpu>(Shape2(1, _wordDim), d_zero);

/*				unit_att_after.ComputeForwardScore(input_after, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_middle[middleSize-1],
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after);*/
				unit_att_after.ComputeForwardScore(input_after, input_after,
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after);

				// input -> hidden
				combine_hidden_layer.ComputeForwardScore(y_before[beforeSize-1], y_att_before,
						y_entityFormer[enFormerSize-1], y_middle[middleSize-1], y_att_middle,
						y_entityLatter[enLatterSize-1], y_after[afterSize-1], y_att_after,
						hidden);

				// hidden -> output
				output_layer.ComputeForwardScore(hidden, output);

				// get delta for each output
				cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

				// loss backward propagation
				// output
				output_layer.ComputeBackwardLoss(hidden, output, outputLoss, hiddenLoss);

				// hidden
				combine_hidden_layer.ComputeBackwardLoss(
						y_before[beforeSize-1], y_att_before,
						y_entityFormer[enFormerSize-1], y_middle[middleSize-1], y_att_middle,
						y_entityLatter[enLatterSize-1], y_after[afterSize-1], y_att_after,
						hidden, hiddenLoss,
						loss_before[beforeSize-1], ly_att_before,
						loss_entityFormer[enFormerSize-1], loss_middle[middleSize-1] , ly_att_middle,
						loss_entityLatter[enLatterSize-1], loss_after[afterSize-1],ly_att_after);

				// attention
/*				unit_att_before.ComputeBackwardLoss(input_before, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_middle[middleSize-1], y_after[afterSize-1],
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before,
				      ly_att_before, inputLoss_before,
					  loss_entityFormer[enFormerSize-1], loss_entityLatter[enLatterSize-1],
					  loss_middle[middleSize-1], loss_after[afterSize-1]);

				unit_att_middle.ComputeBackwardLoss(input_middle, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_after[afterSize-1],
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle,
						ly_att_middle, inputLoss_middle,
						loss_entityFormer[enFormerSize-1], loss_entityLatter[enLatterSize-1],
						loss_before[beforeSize-1], loss_after[afterSize-1]);

				unit_att_after.ComputeBackwardLoss(input_after, y_entityFormer[enFormerSize-1],
						y_entityLatter[enLatterSize-1], y_before[beforeSize-1], y_middle[middleSize-1],
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after,
						ly_att_after, inputLoss_after,
						loss_entityFormer[enFormerSize-1], loss_entityLatter[enLatterSize-1],
						loss_before[beforeSize-1], loss_middle[middleSize-1]);*/

				unit_att_before.ComputeBackwardLoss(input_before, input_before,
						xMExp_before, xExp_before, xSum_before,
						xPoolIndex_before, y_att_before,
						ly_att_before, inputLoss_before, inputLoss_before);

				unit_att_middle.ComputeBackwardLoss(input_middle, input_middle,
						xMExp_middle, xExp_middle, xSum_middle,
						xPoolIndex_middle, y_att_middle,
						ly_att_middle, inputLoss_middle, inputLoss_middle);

				unit_att_after.ComputeBackwardLoss(input_after, input_after,
						xMExp_after, xExp_after, xSum_after,
						xPoolIndex_after, y_att_after,
						ly_att_after, inputLoss_after, inputLoss_after);


				// release
				for (int idx = 0; idx < beforeSize; idx++) {
					FreeSpace(&(xMExp_before[idx]));
					FreeSpace(&(xExp_before[idx]));
					FreeSpace(&(xPoolIndex_before[idx]));
				}
				FreeSpace(&(xSum_before));
				FreeSpace(&(y_att_before));
				FreeSpace(&(ly_att_before));

				for (int idx = 0; idx < middleSize; idx++) {
					FreeSpace(&(xMExp_middle[idx]));
					FreeSpace(&(xExp_middle[idx]));
					FreeSpace(&(xPoolIndex_middle[idx]));
				}
				FreeSpace(&(xSum_middle));
				FreeSpace(&(y_att_middle));
				FreeSpace(&(ly_att_middle));

				for (int idx = 0; idx < afterSize; idx++) {
					FreeSpace(&(xMExp_after[idx]));
					FreeSpace(&(xExp_after[idx]));
					FreeSpace(&(xPoolIndex_after[idx]));
				}
				FreeSpace(&(xSum_after));
				FreeSpace(&(y_att_after));
				FreeSpace(&(ly_att_after));

			} else {
				// input -> hidden
				hidden_layer.ComputeForwardScore(y_before[beforeSize-1],
					y_entityFormer[enFormerSize-1], y_middle[middleSize-1], y_entityLatter[enLatterSize-1],
					y_after[afterSize-1], hidden);

				// hidden -> output
				output_layer.ComputeForwardScore(hidden, output);

				// get delta for each output
				cost += softmax_loss(output, example.m_labels, outputLoss, _eval, example_num);

				// loss backward propagation
				// output
				output_layer.ComputeBackwardLoss(hidden, output, outputLoss, hiddenLoss);

				// hidden
				hidden_layer.ComputeBackwardLoss(y_before[beforeSize-1],
						y_entityFormer[enFormerSize-1], y_middle[middleSize-1],
						y_entityLatter[enLatterSize-1], y_after[afterSize-1],
						hidden, hiddenLoss,
						loss_before[beforeSize-1],
					loss_entityFormer[enFormerSize-1], loss_middle[middleSize-1],
					loss_entityLatter[enLatterSize-1],
					loss_after[afterSize-1]);


			}


			// input
			unit_before.ComputeBackwardLoss(input_before, iy_before, oy_before,
							      fy_before, mcy_before, cy_before, my_before,
							      y_before, loss_before, inputLoss_before);
			unit_middle.ComputeBackwardLoss(input_middle, iy_middle, oy_middle,
							      fy_middle, mcy_middle, cy_middle, my_middle,
							      y_middle, loss_middle, inputLoss_middle);
			unit_after.ComputeBackwardLoss(input_after, iy_after, oy_after,
							      fy_after, mcy_after, cy_after, my_after,
							      y_after, loss_after, inputLoss_after);

			unit_entityFormer.ComputeBackwardLoss(input_entityFormer, iy_entityFormer, oy_entityFormer,
				      fy_entityFormer, mcy_entityFormer, cy_entityFormer, my_entityFormer,
				      y_entityFormer, loss_entityFormer, inputLoss_entityFormer);

			unit_entityLatter.ComputeBackwardLoss(input_entityLatter, iy_entityLatter, oy_entityLatter,
							      fy_entityLatter, mcy_entityLatter, cy_entityLatter, my_entityLatter,
							      y_entityLatter, loss_entityLatter, inputLoss_entityLatter);


			// word
			if (_words.bEmbFineTune()) {
				for (int idx = 0; idx < beforeSize; idx++) {
					inputLoss_before[idx] = inputLoss_before[idx] * mask_before[idx];
					_words.EmbLoss(example.m_before[idx], inputLoss_before[idx]);
				}
				for (int idx = 0; idx < enFormerSize; idx++) {
					inputLoss_entityFormer[idx] = inputLoss_entityFormer[idx] * mask_entityFormer[idx];
					_words.EmbLoss(example.m_entityFormer[idx], inputLoss_entityFormer[idx]);
				}
				for (int idx = 0; idx < enLatterSize; idx++) {
					inputLoss_entityLatter[idx] = inputLoss_entityLatter[idx] * mask_entityLatter[idx];
					_words.EmbLoss(example.m_entityLatter[idx], inputLoss_entityLatter[idx]);
				}
				for (int idx = 0; idx < middleSize; idx++) {
					inputLoss_middle[idx] = inputLoss_middle[idx] * mask_middle[idx];
					_words.EmbLoss(example.m_middle[idx], inputLoss_middle[idx]);
				}
				for (int idx = 0; idx < afterSize; idx++) {
					inputLoss_after[idx] = inputLoss_after[idx] * mask_after[idx];
					_words.EmbLoss(example.m_after[idx], inputLoss_after[idx]);
				}
			}

			//release all the stuff
			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(input_before[idx]));
				FreeSpace(&(mask_before[idx]));
				FreeSpace(&(inputLoss_before[idx]));
			}
			for (int idx = 0; idx < enFormerSize; idx++) {
				FreeSpace(&(input_entityFormer[idx]));
				FreeSpace(&(mask_entityFormer[idx]));
				FreeSpace(&(inputLoss_entityFormer[idx]));
			}

			for (int idx = 0; idx < enLatterSize; idx++) {
				FreeSpace(&(input_entityLatter[idx]));
				FreeSpace(&(mask_entityLatter[idx]));
				FreeSpace(&(inputLoss_entityLatter[idx]));
			}

			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(input_middle[idx]));
				FreeSpace(&(mask_middle[idx]));
				FreeSpace(&(inputLoss_middle[idx]));
			}

			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(input_after[idx]));
				FreeSpace(&(mask_after[idx]));
				FreeSpace(&(inputLoss_after[idx]));
			}

			FreeSpace(&hidden);
			FreeSpace(&hiddenLoss);

			FreeSpace(&output);
			FreeSpace(&outputLoss);

			for (int idx = 0; idx < beforeSize; idx++) {
				FreeSpace(&(iy_before[idx]));
				FreeSpace(&(oy_before[idx]));
				FreeSpace(&(fy_before[idx]));
				FreeSpace(&(mcy_before[idx]));
				FreeSpace(&(cy_before[idx]));
				FreeSpace(&(my_before[idx]));
				FreeSpace(&(y_before[idx]));
				FreeSpace(&(loss_before[idx]));
			}

			for (int idx = 0; idx < enFormerSize; idx++) {
				FreeSpace(&(iy_entityFormer[idx]));
				FreeSpace(&(oy_entityFormer[idx]));
				FreeSpace(&(fy_entityFormer[idx]));
				FreeSpace(&(mcy_entityFormer[idx]));
				FreeSpace(&(cy_entityFormer[idx]));
				FreeSpace(&(my_entityFormer[idx]));
				FreeSpace(&(y_entityFormer[idx]));
				FreeSpace(&(loss_entityFormer[idx]));
			}

			for (int idx = 0; idx < enLatterSize; idx++) {
				FreeSpace(&(iy_entityLatter[idx]));
				FreeSpace(&(oy_entityLatter[idx]));
				FreeSpace(&(fy_entityLatter[idx]));
				FreeSpace(&(mcy_entityLatter[idx]));
				FreeSpace(&(cy_entityLatter[idx]));
				FreeSpace(&(my_entityLatter[idx]));
				FreeSpace(&(y_entityLatter[idx]));
				FreeSpace(&(loss_entityLatter[idx]));
			}

			for (int idx = 0; idx < middleSize; idx++) {
				FreeSpace(&(iy_middle[idx]));
				FreeSpace(&(oy_middle[idx]));
				FreeSpace(&(fy_middle[idx]));
				FreeSpace(&(mcy_middle[idx]));
				FreeSpace(&(cy_middle[idx]));
				FreeSpace(&(my_middle[idx]));
				FreeSpace(&(y_middle[idx]));
				FreeSpace(&(loss_middle[idx]));
			}

			for (int idx = 0; idx < afterSize; idx++) {
				FreeSpace(&(iy_after[idx]));
				FreeSpace(&(oy_after[idx]));
				FreeSpace(&(fy_after[idx]));
				FreeSpace(&(mcy_after[idx]));
				FreeSpace(&(cy_after[idx]));
				FreeSpace(&(my_after[idx]));
				FreeSpace(&(y_after[idx]));
				FreeSpace(&(loss_after[idx]));
			}

		} // end for example_num


		return cost;
	}

	void updateParams() {
		unit_before.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		unit_entityFormer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		unit_entityLatter.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		unit_middle.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		unit_after.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		if(options.input_represent == 2) {
			unit_att_before.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_att_middle.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_att_after.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			combine_hidden_layer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		} else if(options.input_represent == 1) {
			unit_att_before.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_att_middle.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			unit_att_after.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
			hidden_layer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		} else {
			hidden_layer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
		}
		output_layer.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);

		_words.updateAdaGrad(options.regParameter, options.adaAlpha, options.adaEps);
	}

};



#endif /* CLASSIFIER_H_ */
