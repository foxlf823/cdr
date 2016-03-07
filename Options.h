#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3L.h"

using namespace std;

class Options {
public:

  int wordCutOff;

  dtype initRange;
  int maxIter;
  int batchSize;
  dtype adaEps;
  dtype adaAlpha;
  dtype regParameter;
  dtype dropProb;

  int evalPerIter;

  bool wordEmbFineTune;

  string abbrPath;
  string puncPath;

  int wordEmbSize;
  int entity_embsize;
  int context_embsize;
  int hiddenSize;

  int sent_window;

  int verboseIter;

  string output;
  string embFile;

  bool attention;

  string wordnet;
  string brown;

  int channelMode; // 000001-word, 000010-wordnet, 000100-brown, 001000-bigram, 010000-pos, 100000-sst

  Options() {
    wordCutOff = 0;
    initRange = 0.01;
    maxIter = 1000;
    batchSize = 1;
    adaEps = 1e-6;
    adaAlpha = 0.01;
    regParameter = 1e-8;
    dropProb = 0.5;

    wordEmbSize = 50;
    entity_embsize = 50;
    context_embsize = 100;
    hiddenSize = 150;

    evalPerIter = 1;
    wordEmbFineTune = true;

    abbrPath = "";
    puncPath = "";

    sent_window = 1;
    verboseIter = 0;

    output = "";
    embFile = "";

    attention = false;

    wordnet = "";
    brown = "";

    channelMode = 1;
  }

  Options(const Options& options) {
	  wordCutOff = options.wordCutOff;
	  initRange = options.initRange;
	  maxIter = options.maxIter;
	  batchSize = options.batchSize;
	  adaEps = options.adaEps;
	  adaAlpha = options.adaAlpha;
	  regParameter = options.regParameter;
	  dropProb = options.dropProb;


	  wordEmbSize = options.wordEmbSize;
		entity_embsize = options.entity_embsize;
		context_embsize = options.context_embsize;
		hiddenSize = options.hiddenSize;

	  evalPerIter = options.evalPerIter;
	  wordEmbFineTune = options.wordEmbFineTune;

		abbrPath = options.abbrPath;
		puncPath = options.puncPath;

		sent_window  = options.sent_window;

		verboseIter = options.verboseIter;

		output = options.output;
		embFile = options.embFile;

		attention = options.attention;

		wordnet = options.wordnet;
		brown = options.brown;

		channelMode = options.channelMode;
  }

/*  virtual ~Options() {

  }*/

  void setOptions(const vector<string> &vecOption) {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff")
        wordCutOff = atoi(pr.second.c_str());
      else if (pr.first == "initRange")
        initRange = atof(pr.second.c_str());
      else if (pr.first == "maxIter")
        maxIter = atoi(pr.second.c_str());
      else if (pr.first == "batchSize")
        batchSize = atoi(pr.second.c_str());
      else if (pr.first == "adaEps")
        adaEps = atof(pr.second.c_str());
      else if (pr.first == "adaAlpha")
        adaAlpha = atof(pr.second.c_str());
      else if (pr.first == "regParameter")
        regParameter = atof(pr.second.c_str());
      else if (pr.first == "dropProb")
        dropProb = atof(pr.second.c_str());

      else if (pr.first == "hiddenSize")
        hiddenSize = atoi(pr.second.c_str());
      else if (pr.first == "wordEmbSize")
        wordEmbSize = atoi(pr.second.c_str());
      else if (pr.first == "entity_embsize")
    	  entity_embsize = atoi(pr.second.c_str());
      else if (pr.first == "context_embsize")
    	  context_embsize = atoi(pr.second.c_str());
        

      else if(pr.first == "evalPerIter")
    	  evalPerIter = atoi(pr.second.c_str());

      else if (pr.first == "wordEmbFineTune")
    	  wordEmbFineTune = (pr.second == "true") ? true : false;

      else if(pr.first == "abbrPath")
    	  abbrPath = pr.second;

      else if(pr.first == "puncPath")
    	  puncPath = pr.second;
      else if (pr.first == "sent_window")
    	  sent_window = atoi(pr.second.c_str());
      else if(pr.first == "verboseIter")
    	  verboseIter = atoi(pr.second.c_str());

      else if(pr.first == "output")
          	  output = pr.second;
      else if(pr.first == "embFile")
          embFile = pr.second;

      else if(pr.first == "input_represent")
    	  attention = (pr.second == "true") ? true : false;
      else if(pr.first == "wordnet")
    	  wordnet = pr.second;
      else if(pr.first == "brown")
    	  brown = pr.second;
      else if(pr.first== "channelMode")
    	  channelMode = atoi(pr.second.c_str());
    }
  }

  void showOptions() {
    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "initRange = " << initRange << std::endl;
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;

    std::cout << "hiddenSize = " << hiddenSize << std::endl;
    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout << "entity_embsize = " << entity_embsize << std::endl;
    std::cout << "context_embsize = " << context_embsize << std::endl;

    cout<< "evalPerIter = " << evalPerIter << endl;
    cout<< "wordEmbFineTune = "<<wordEmbFineTune<<endl;

    cout<<"abbrPath = "<<abbrPath<<endl;
    cout<<"puncPath = "<<puncPath<<endl;

    cout<<"sent_window = "<<sent_window<<endl;
    cout<<"verboseIter = "<<verboseIter<<endl;
    cout<<"output = "<<output<<endl;
    cout<<"embFile = "<<embFile<<endl;

	cout<<"attention = "<<attention<<endl;

	cout<<"wordnet = "<<wordnet<<endl;
	cout<<"brown = "<<brown<<endl;
	cout<<"channelMode = "<<channelMode<<endl;
  }

  void load(const std::string& infile) {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(inf, strLine)) {
        break;
      }
      if (strLine.empty())
        continue;
      vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }
};

#endif

