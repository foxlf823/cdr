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
	string otherDir; // use other to supplement training data
	string predictTestFile;
	string trainNlpFile;
	string devNlpFile;
	string testNlpFile;
	bool usedev = false;
	string trainSstFile;
	string devSstFile;
	string testSstFile;


	dsr::Argument_helper ah;
	ah.new_named_string("train", "", "", "", trainFile);
	ah.new_named_string("dev", "", "", "", devFile);
	ah.new_named_string("test", "", "", "", testFile);
	ah.new_named_string("other", "", "", "", otherDir);
	ah.new_named_string("option", "", "", "", optionFile);
	ah.new_named_string("output", "", "", "", outputFile);
	ah.new_flag("usedev", "", "", usedev);
	ah.new_named_string("predict", "", "", "", predictTestFile);
	ah.new_named_string("trainnlp", "", "", "", trainNlpFile);
	ah.new_named_string("devnlp", "", "", "", devNlpFile);
	ah.new_named_string("testnlp", "", "", "", testNlpFile);
	ah.new_named_string("trainsst", "", "", "", trainSstFile);
	ah.new_named_string("devsst", "", "", "", devSstFile);
	ah.new_named_string("testsst", "", "", "", testSstFile);
	ah.process(argc, argv);
	cout<<"train file: " <<trainFile <<endl;
	cout<<"dev file: "<<devFile<<endl;
	cout<<"test file: "<<testFile<<endl;
	cout<<"other Dir: "<<otherDir<<endl;
	cout<<"usedev: "<<usedev<<endl;
	cout<<"predict test file: "<<predictTestFile<<endl;
	cout<<"trainnlp file: "<<trainNlpFile<<endl;
	cout<<"devnlp file: "<<devNlpFile<<endl;
	cout<<"testnlp file: "<<testNlpFile<<endl;
	cout<<"trainsst file: "<<trainSstFile<<endl;
	cout<<"devsst file: "<<devSstFile<<endl;
	cout<<"testsst file: "<<testSstFile<<endl;

	Options options;
	options.load(optionFile);

	options.output = outputFile;

	options.showOptions();

	Tool tool(options);

	NNcdr nncdr(options);

	nncdr.train(trainFile, devFile, testFile, otherDir, tool, usedev, predictTestFile,
			trainNlpFile, devNlpFile, testNlpFile, trainSstFile, devSstFile, testSstFile);


#if USE_CUDA==1
  ShutdownTensorEngine();
#else
  ShutdownTensorEngine<cpu>();
#endif

    return 0;

}

