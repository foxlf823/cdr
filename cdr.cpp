/*
 * cdr.cpp
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#include <vector>
#include "BiocDocument.h"
#include "utils.h"
#include "FoxUtil.h"
#include <iostream>
#include "Token.h"
#include "SentSplitter.h"
#include "Tokenizer.h"
#include "N3L.h"
#include "Argument_helper.h"
#include "Options.h"
#include "NNcdr.h"

using namespace std;


int main(int argc, char **argv)
{
#if USE_CUDA==1
  InitTensorEngine();
#else
  InitTensorEngine<cpu>();
#endif

	string optionFile;
	string trainFile;
	string devFile;
	string testFile;
	string outputFile;
	//bool debug = false;

	dsr::Argument_helper ah;
	ah.new_named_string("train", "", "", "", trainFile);
	ah.new_named_string("dev", "", "", "", devFile);
	ah.new_named_string("test", "", "", "", testFile);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("output", "", "", "", outputFile);
	//ah.new_flag("d", "debug", "debug", debug);
	ah.process(argc, argv);
	cout<<"train file: " <<trainFile <<endl;
	cout<<"dev file: "<<devFile<<endl;
	cout<<"test file: "<<testFile<<endl;

	Options options;
	options.load(optionFile);

	options.output = outputFile;

	options.showOptions();

	Tool tool(options);

	NNcdr nncdr(options);

	nncdr.train(trainFile, devFile, testFile, tool);


#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif

    return 0;

}

