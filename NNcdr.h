/*
 * NNcdr.h
 *
 *  Created on: Dec 27, 2015
 *      Author: fox
 */

#ifndef NNCDR_H_
#define NNCDR_H_

#include "Options.h"
#include "Tool.h"
#include "FoxUtil.h"
#include "N3Lhelper.h"
#include "Utf.h"
#include "Classifier.h"
#include "Token.h"
#include "Sent.h"
#include <sstream>
#include "N3L.h"
#include "wnb/core/wordnet.hh"
#include "Document.h"
#include "EnglishPos.h"
#include "Punctuation.h"

using namespace nr;
using namespace std;
using namespace wnb;
using namespace fox;

// a nn model to classify the cdr relation
class NNcdr {
public:
	Options m_options;
	Alphabet m_wordAlphabet;
	Alphabet m_wordnetAlphabet;
	Alphabet m_brownAlphabet;
	Alphabet m_bigramAlphabet;
	Alphabet m_posAlphabet;
	Alphabet m_sstAlphabet;

	string nullkey;
	string unknownkey;
	string chemicalkey;
	string diseasekey;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
  Classifier<cpu> m_classifier;
#endif

	NNcdr(const Options &options):m_options(options), m_classifier(options) {
		nullkey = "-#null#-";
		unknownkey = "-#unknown#-";
		chemicalkey = "-chem-";
		diseasekey = "-dise-";
	}
/*
	virtual ~NNcdr() {

	}
*/

	// return the best choice
	bool predict(const Example& example) {
		vector<double> scores(2);
		m_classifier.predict(example, scores);

		return scores[0]>scores[1] ? true:false;
	}

	void evaluate(const Example& example, bool predicted, Metric& eval) {
		bool gold = example.m_labels[0]>example.m_labels[1] ? true:false;

/*		if(gold==true)
			eval.overall_label_count ++;
		if(predicted==true)
			eval.predicated_label_count ++;

		if(gold==true && predicted==true)
			eval.correct_label_count++;*/

		eval.overall_label_count ++;
		//eval.predicated_label_count++;
		if(gold==predicted)
			eval.correct_label_count++;
	}

	void train(const string& trainFile, const string& devFile, const string& testFile, const string& otherDir,
			Tool& tool, bool usedev, const string& predictTestFile,
			const string& trainNlpFile, const string& devNlpFile, const string& testNlpFile,
			const string& trainSstFile, const string& devSstFile, const string& testSstFile) {


		// load train data
		vector<BiocDocument> trainDocuments;
		parseBioc(trainFile, trainDocuments);
		vector<BiocDocument> devDocuments;
		if(!devFile.empty()) {
			parseBioc(devFile, devDocuments);
		}
		vector<BiocDocument> testDocuments;
		if(!testFile.empty()) {
			parseBioc(testFile, testDocuments);
			if(!predictTestFile.empty()) {
				vector<BiocDocument> predictTestDocuments;
				parseBioc(predictTestFile, predictTestDocuments);
				// replace gold entities with predicted
				for(int i=0;i<predictTestDocuments.size();i++) {
					testDocuments[i].entities.clear();
					testDocuments[i].entities = predictTestDocuments[i].entities;
				}
			}
		}
		vector<BiocDocument> otherDocuments;
		if(!otherDir.empty()) {
			parseBiocDir(otherDir, otherDocuments);
			// add otherDocuments to trainDocuments
			/*for(int i=0;i<otherDocuments.size();i++) {
				trainDocuments.push_back(otherDocuments[i]);
			}*/
		}

		vector<Document> trainNlpdocs;
		if(!trainNlpFile.empty()) {
			loadNlpFile(trainNlpFile, trainNlpdocs);
			if(!trainSstFile.empty()) {
				loadSstFile(trainSstFile, trainNlpdocs);
			}
		}

		vector<Document> devNlpdocs;
		if(!devNlpFile.empty()) {
			loadNlpFile(devNlpFile, devNlpdocs);
			if(!devSstFile.empty()) {
				loadSstFile(devSstFile, devNlpdocs);
			}
		}

		vector<Document> testNlpdocs;
		if(!testNlpFile.empty()) {
			loadNlpFile(testNlpFile, testNlpdocs);
			if(!testSstFile.empty()) {
				loadSstFile(testSstFile, testNlpdocs);
			}
		}




		if(usedev) {
			for(int i=0;i<devDocuments.size();i++) {
				trainDocuments.push_back(devDocuments[i]);
				trainNlpdocs.push_back(devNlpdocs[i]);
			}
			devDocuments.clear();
			devNlpdocs.clear();
			for(int i=0;i<testDocuments.size();i++) {
				devDocuments.push_back(testDocuments[i]);
				devNlpdocs.push_back(testNlpdocs[i]);
			}
		}


		cout << "Creating Alphabet..." << endl;
		/*
		 * For all alphabets, unknownkey and nullkey should be 0 and 1.
		 */
		m_wordAlphabet.clear();
		m_wordAlphabet.from_string(unknownkey);
		m_wordAlphabet.from_string(nullkey);
		m_wordAlphabet.from_string(chemicalkey);
		m_wordAlphabet.from_string(diseasekey);

		m_wordnetAlphabet.clear();
		m_wordnetAlphabet.from_string(unknownkey);
		m_wordnetAlphabet.from_string(nullkey);

		m_brownAlphabet.clear();
		m_brownAlphabet.from_string(unknownkey);
		m_brownAlphabet.from_string(nullkey);

		m_bigramAlphabet.clear();
		m_bigramAlphabet.from_string(unknownkey);
		m_bigramAlphabet.from_string(nullkey);

		m_posAlphabet.clear();
		m_posAlphabet.from_string(unknownkey);
		m_posAlphabet.from_string(nullkey);

		m_sstAlphabet.clear();
		m_sstAlphabet.from_string(unknownkey);
		m_sstAlphabet.from_string(nullkey);

		createAlphabet(trainDocuments, tool, trainNlpdocs);
		if(!otherDir.empty()) {
			vector<Document> otherNlpdocs;
			createAlphabet(otherDocuments, tool, otherNlpdocs);
		}




		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devDocuments.empty())
				createAlphabet(devDocuments, tool, devNlpdocs);
			if(!testDocuments.empty())
				createAlphabet(testDocuments, tool, testNlpdocs);
		}

		NRMat<dtype> wordEmb;
		if(m_options.wordEmbFineTune) {
			if(m_options.embFile.empty()) {
				cout<<"random emb"<<endl;

				randomInitNrmat(wordEmb, m_wordAlphabet);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v.loadFromBinFile(m_options.embFile, true);
				// format the words of pre-trained embeddings
				formatWords(tool.w2v);
				double emb[m_wordAlphabet.size()][m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);

				tool.w2v.getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);


			}
		} else {
			if(m_options.embFile.empty()) {
				assert(0);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v.loadFromBinFile(m_options.embFile, true);
				// format the words of pre-trained embeddings
				formatWords(tool.w2v);
				double emb[m_wordAlphabet.size()][m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);

				tool.w2v.getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);
			}
		}

		NRMat<dtype> wordnetEmb;
		randomInitNrmat(wordnetEmb, m_wordnetAlphabet);
		NRMat<dtype> brownEmb;
		randomInitNrmat(brownEmb, m_brownAlphabet);
		NRMat<dtype> bigramEmb;
		randomInitNrmat(bigramEmb, m_bigramAlphabet);
		NRMat<dtype> posEmb;
		randomInitNrmat(posEmb, m_posAlphabet);
		NRMat<dtype> sstEmb;
		randomInitNrmat(sstEmb, m_sstAlphabet);

		m_classifier.init(wordEmb, wordnetEmb,brownEmb, bigramEmb, posEmb, sstEmb);

		vector<Example> trainExamples;
		initialExamples(tool, trainDocuments, trainExamples,true, false, trainNlpdocs);
		if(!otherDir.empty()) {
			vector<Example> otherExamples;
			vector<Document> otherNlpDocs;
			initialExamples(tool, otherDocuments, otherExamples,true, true, otherNlpDocs);
			//cout<<"Total other example number: "<<otherExamples.size()<<endl;
			for(int i=0;i<otherExamples.size();i++) {
				//if(otherExamples[i].m_labels[0]==1)
					trainExamples.push_back(otherExamples[i]);
			}
		}
		cout<<"Total train example number: "<<trainExamples.size()<<endl;
		vector<Example> devExamples;
		if(!devFile.empty()) {
			initialExamples(tool, devDocuments, devExamples,false, false, devNlpdocs);
			cout<<"Total dev example number: "<<devExamples.size()<<endl;
		}
		vector<Example> testExamples;
		if(!testFile.empty()) {
			initialExamples(tool, testDocuments, testExamples,false, false, testNlpdocs);
			cout<<"Total test example number: "<<testExamples.size()<<endl;
		}


		int inputSize = trainExamples.size();
		int batchBlock = inputSize / m_options.batchSize;
		if (inputSize % m_options.batchSize != 0)
			batchBlock++;

		std::vector<int> indexes;
		for (int i = 0; i < inputSize; ++i)
			indexes.push_back(i);

		static Metric eval, metric_dev;
		static vector<Example> subExamples;
		int devNum = devExamples.size(), testNum = testExamples.size();

		dtype best = 0;
		vector<Example> toBeOutput;

		// begin to train
		for (int iter = 0; iter < m_options.maxIter; ++iter) {
			if(m_options.verboseIter>0)
				cout << "##### Iteration " << iter << std::endl;

		    random_shuffle(indexes.begin(), indexes.end());
		    eval.reset();

		    // use all batches to train during an iteration
		    for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
				subExamples.clear();
				int start_pos = updateIter * m_options.batchSize;
				int end_pos = (updateIter + 1) * m_options.batchSize;
				if (end_pos > inputSize)
					end_pos = inputSize;

				for (int idy = start_pos; idy < end_pos; idy++) {
					subExamples.push_back(trainExamples[indexes[idy]]);
				}

				int curUpdateIter = iter * batchBlock + updateIter;
				dtype cost = m_classifier.process(subExamples, curUpdateIter);

				eval.overall_label_count += m_classifier._eval.overall_label_count;
				eval.correct_label_count += m_classifier._eval.correct_label_count;

		      if (m_options.verboseIter>0 && (curUpdateIter + 1) % m_options.verboseIter == 0) {
		        //m_classifier.checkgrads(subExamples, curUpdateIter+1);
		        //std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
		        std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
		      }
		      m_classifier.updateParams();

		    }

		    // an iteration end, begin to evaluate
		    if (devExamples.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words);
		    		averageUnkownEmb(m_wordnetAlphabet, m_classifier._wordnet);
		    		averageUnkownEmb(m_brownAlphabet, m_classifier._brown);
		    		averageUnkownEmb(m_bigramAlphabet, m_classifier._bigram);
		    		averageUnkownEmb(m_posAlphabet, m_classifier._pos);
		    		averageUnkownEmb(m_sstAlphabet, m_classifier._sst);
		    	}

		    	metric_dev.reset();
		    	if(usedev) {
		    		toBeOutput.clear();
					for (int idx = 0; idx < testExamples.size(); idx++) {
						bool predicted = predict(testExamples[idx]);
						evaluate(testExamples[idx], predicted, metric_dev);
						if(predicted) {
							toBeOutput.push_back(testExamples[idx]);
						}
					}
					metric_dev.print();
			    	if (metric_dev.getAccuracy() > best) {
			    		cout << "Exceeds best performance of " << best << endl;
			    		best = metric_dev.getAccuracy();
			    		outputToPubtator(toBeOutput, m_options.output);
			    	}
		    	} else {
			    	for (int idx = 0; idx < devExamples.size(); idx++) {
			    		bool predicted = predict(devExamples[idx]);
						evaluate(devExamples[idx], predicted, metric_dev);
			    	}
			    	metric_dev.print();
			    	if (metric_dev.getAccuracy() > best) {
			    		cout << "Exceeds best performance of " << best << endl;
			    		best = metric_dev.getAccuracy();
			    		// if the current exceeds the best, we do the blind test on the test set
			    		// but don't evaluate and store the results for the official evaluation
						if (testExamples.size() > 0) {
							toBeOutput.clear();

							for (int idx = 0; idx < testExamples.size(); idx++) {
								bool predicted = predict(testExamples[idx]);


								if(predicted) {
									toBeOutput.push_back(testExamples[idx]);
								}
							}

							outputToPubtator(toBeOutput, m_options.output);
						}
			    	}
		    	}






		    } // devExamples > 0

		} // for iter




		m_classifier.release();

	}

	void initialExamples(Tool& tool, const vector<BiocDocument>& documents, vector<Example>& examples,
			bool bStatistics, bool bOther, const vector<Document>& nlpdocs) {
		int ctPositive = 0;
		int ctNegtive = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			int ctPositiveInCurrentDoc =0;
			int ctNegtiveInCurrentDoc = 0;

			//cout<<documents[docIdx].id<<endl;
			string text = documents[docIdx].title+" "+documents[docIdx].abstract;
			vector<string> str_sentences;
			tool.sentSplitter.splitWithFilters(text, str_sentences);
			vector<fox::Sent> sents;
			int offset = 0;
			for(int sentIdx=0;sentIdx<str_sentences.size();sentIdx++) {
				fox::Sent sent;
				tool.tokenizer.tokenize(offset, str_sentences[sentIdx],sent.tokens);
				sent.begin = offset;
				sent.end = offset+(str_sentences[sentIdx]).length();
				sents.push_back(sent);

				// find all the entities in the current sentence
				vector<Entity> Bentity;
				findEntityInSent(sent.begin, sent.end, documents[docIdx], Bentity);
				/*
				 * for each entity B, scan all the entities A before it but in the sent_window
				 * so A is before B and their type should not be the same
				 */
				int windowBegin = sentIdx-m_options.sent_window+1 >=0 ? sentIdx-m_options.sent_window+1 : 0;
				for(int b=0;b<Bentity.size();b++) {
					if(Bentity[b].mesh == "-1")
						continue;
					/*else if(Bentity[b].compositeRole=="CompositeMention")
						continue;*/
					else if(Bentity[b].compositeRole=="IndividualMention")
						continue;


					for(int i=windowBegin;i<=sentIdx;i++) {
						vector<Entity> Aentity;
						findEntityInSent(sents[i].begin, sents[i].end, documents[docIdx], Aentity);
						for(int a=0;a<Aentity.size();a++) {

							if(Aentity[a].mesh == "-1")
								continue;
							/*else if(Aentity[a].compositeRole=="CompositeMention")
								continue;*/
							else if(Aentity[a].compositeRole=="IndividualMention")
								continue;
							else if(Aentity[a].begin >= Bentity[b].begin)
								continue;
							else if(Aentity[a].type == Bentity[b].type)
								continue;


							Example eg;
							if(isADE(Aentity[a], Bentity[b], documents[docIdx])) {
								// positive
								eg.m_labels.push_back(1);
								eg.m_labels.push_back(0);

							} else {
								// negative
								eg.m_labels.push_back(0);
								eg.m_labels.push_back(1);

							}

							Entity& chemical = Aentity[a].type=="Chemical" ? Aentity[a]:Bentity[b];
							Entity& disease = Aentity[a].type=="Disease" ? Aentity[a]:Bentity[b];

							for(int j=i;j<=sentIdx;j++) {
								for(int k=0;k<sents[j].tokens.size();k++) {
									if(isTokenBeforeEntity(sents[j].tokens[k], Aentity[a])) {

										if(!isTokenSatisfied(nlpdocs[docIdx].sentences[j].tokens[k], sents[j].tokens[k]))
											continue;


										featureName2ID(m_wordAlphabet, feature_word(sents[j].tokens[k].word), eg.m_before);
										featureName2ID(m_wordnetAlphabet, feature_wordnet(sents[j].tokens[k].word, tool), eg.m_before_wordnet);
										featureName2ID(m_brownAlphabet, feature_brown(sents[j].tokens[k].word, tool), eg.m_before_brown);
										featureName2ID(m_bigramAlphabet, feature_bigram(sents[j].tokens, k, tool), eg.m_before_bigram);
										featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_before_pos);
										featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_before_sst);
									}
									else if(isTokenAfterEntity(sents[j].tokens[k], Bentity[b])) {
										if(!isTokenSatisfied(nlpdocs[docIdx].sentences[j].tokens[k], sents[j].tokens[k]))
											continue;

										featureName2ID(m_wordAlphabet, feature_word(sents[j].tokens[k].word), eg.m_after);
										featureName2ID(m_wordnetAlphabet, feature_wordnet(sents[j].tokens[k].word, tool), eg.m_after_wordnet);
										featureName2ID(m_brownAlphabet, feature_brown(sents[j].tokens[k].word, tool), eg.m_after_brown);
										featureName2ID(m_bigramAlphabet, feature_bigram(sents[j].tokens, k, tool), eg.m_after_bigram);
										featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_after_pos);
										featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_after_sst);
									}
									else if(isTokenInEntity(sents[j].tokens[k], chemical)) {
										featureName2ID(m_wordAlphabet, feature_word(sents[j].tokens[k].word), eg.m_entityFormer);
										featureName2ID(m_wordnetAlphabet, feature_wordnet(sents[j].tokens[k].word, tool), eg.m_entityFormer_wordnet);
										featureName2ID(m_brownAlphabet, feature_brown(sents[j].tokens[k].word, tool), eg.m_entityFormer_brown);
										featureName2ID(m_bigramAlphabet, feature_bigram(sents[j].tokens, k, tool), eg.m_entityFormer_bigram);
										featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_entityFormer_pos);
										featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_entityFormer_sst);
									} else if(isTokenInEntity(sents[j].tokens[k], disease)) {
										featureName2ID(m_wordAlphabet, feature_word(sents[j].tokens[k].word), eg.m_entityLatter);
										featureName2ID(m_wordnetAlphabet, feature_wordnet(sents[j].tokens[k].word, tool), eg.m_entityLatter_wordnet);
										featureName2ID(m_brownAlphabet, feature_brown(sents[j].tokens[k].word, tool), eg.m_entityLatter_brown);
										featureName2ID(m_bigramAlphabet, feature_bigram(sents[j].tokens, k, tool), eg.m_entityLatter_bigram);
										featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_entityLatter_pos);
										featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_entityLatter_sst);
									} else if(isTokenBetweenTwoEntities(sents[j].tokens[k], Aentity[a], Bentity[b])){
										if(!isTokenSatisfied(nlpdocs[docIdx].sentences[j].tokens[k], sents[j].tokens[k]))
											continue;

										featureName2ID(m_wordAlphabet, feature_word(sents[j].tokens[k].word), eg.m_middle);
										featureName2ID(m_wordnetAlphabet, feature_wordnet(sents[j].tokens[k].word, tool), eg.m_middle_wordnet);
										featureName2ID(m_brownAlphabet, feature_brown(sents[j].tokens[k].word, tool), eg.m_middle_brown);
										featureName2ID(m_bigramAlphabet, feature_bigram(sents[j].tokens, k, tool), eg.m_middle_bigram);
										featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_middle_pos);
										featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_middle_sst);
									}

								}
							}

							// in case that null
							if(eg.m_before.size()==0) {
								eg.m_before.push_back(m_wordAlphabet.from_string(nullkey));
								eg.m_before_wordnet.push_back(m_wordnetAlphabet.from_string(nullkey));
								eg.m_before_brown.push_back(m_brownAlphabet.from_string(nullkey));
								eg.m_before_bigram.push_back(m_bigramAlphabet.from_string(nullkey));
								eg.m_before_pos.push_back(m_posAlphabet.from_string(nullkey));
								eg.m_before_sst.push_back(m_sstAlphabet.from_string(nullkey));
							}
							if(eg.m_entityFormer.size()==0) {
								eg.m_entityFormer.push_back(m_wordAlphabet.from_string(nullkey));
								eg.m_entityFormer_wordnet.push_back(m_wordnetAlphabet.from_string(nullkey));
								eg.m_entityFormer_brown.push_back(m_brownAlphabet.from_string(nullkey));
								eg.m_entityFormer_bigram.push_back(m_bigramAlphabet.from_string(nullkey));
								eg.m_entityFormer_pos.push_back(m_posAlphabet.from_string(nullkey));
								eg.m_entityFormer_sst.push_back(m_sstAlphabet.from_string(nullkey));
							}
							if(eg.m_entityLatter.size()==0) {
								eg.m_entityLatter.push_back(m_wordAlphabet.from_string(nullkey));
								eg.m_entityLatter_wordnet.push_back(m_wordnetAlphabet.from_string(nullkey));
								eg.m_entityLatter_brown.push_back(m_brownAlphabet.from_string(nullkey));
								eg.m_entityLatter_bigram.push_back(m_bigramAlphabet.from_string(nullkey));
								eg.m_entityLatter_pos.push_back(m_posAlphabet.from_string(nullkey));
								eg.m_entityLatter_sst.push_back(m_sstAlphabet.from_string(nullkey));
							}
							if(eg.m_middle.size()==0) {
								eg.m_middle.push_back(m_wordAlphabet.from_string(nullkey));
								eg.m_middle_wordnet.push_back(m_wordnetAlphabet.from_string(nullkey));
								eg.m_middle_brown.push_back(m_brownAlphabet.from_string(nullkey));
								eg.m_middle_bigram.push_back(m_bigramAlphabet.from_string(nullkey));
								eg.m_middle_pos.push_back(m_posAlphabet.from_string(nullkey));
								eg.m_middle_sst.push_back(m_sstAlphabet.from_string(nullkey));
							}
							if(eg.m_after.size()==0) {
								eg.m_after.push_back(m_wordAlphabet.from_string(nullkey));
								eg.m_after_wordnet.push_back(m_wordnetAlphabet.from_string(nullkey));
								eg.m_after_brown.push_back(m_brownAlphabet.from_string(nullkey));
								eg.m_after_bigram.push_back(m_bigramAlphabet.from_string(nullkey));
								eg.m_after_pos.push_back(m_posAlphabet.from_string(nullkey));
								eg.m_after_sst.push_back(m_sstAlphabet.from_string(nullkey));
							}

							eg.docID = documents[docIdx].id;
							if(Aentity[a].type == "Chemical") {
								eg.chemcalMesh = Aentity[a].mesh;
								eg.diseaseMesh = Bentity[b].mesh;

							} else {
								eg.chemcalMesh = Bentity[b].mesh;
								eg.diseaseMesh = Aentity[a].mesh;
							}

							/* for other documents
							   if it is positive, we add it directly
							   if negative, we add it if positive is less
							*/

							if(eg.m_labels[0]==1) {
								ctPositive++;

								examples.push_back(eg);
								ctPositiveInCurrentDoc++;
							}
							else {
								ctNegtive++;

								if(bOther) {
									if(ctPositiveInCurrentDoc>ctNegtiveInCurrentDoc) {
										examples.push_back(eg);
										ctNegtiveInCurrentDoc++;
									}
								} else {
									examples.push_back(eg);
									ctNegtiveInCurrentDoc++;
								}


							}


						}


					}

				}



				offset += (str_sentences[sentIdx]).length();
			}
		}

		if(bStatistics) {
			cout<<"Positive example: "<<ctPositive<<endl;
			cout<<"Negative example: "<<ctNegtive<<endl;
			cout<<"Proportion +:"<< (ctPositive*1.0)/(ctPositive+ctNegtive)
					<<" , -:"<<(ctNegtive*1.0)/(ctPositive+ctNegtive)<<endl;
		}

	}

	void createAlphabet (const vector<BiocDocument>& documents, Tool& tool, const vector<Document>& nlpdocs) {

		hash_map<string, int> word_stat;
		hash_map<string, int> wordnet_stat;
		hash_map<string, int> brown_stat;
		hash_map<string, int> bigram_stat;
		hash_map<string, int> pos_stat;
		hash_map<string, int> sst_stat;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
			string text = documents[docIdx].title+" "+documents[docIdx].abstract;
			vector<string> sentences;
			tool.sentSplitter.splitWithFilters(text, sentences);

			int offset = 0;
			for(int i=0;i<sentences.size();i++) {
				vector<fox::Token> tokens;
				tool.tokenizer.tokenize(offset, sentences[i],tokens);

				for(int j=0;j<tokens.size();j++) {
					if(!isTokenSatisfied(nlpdocs[docIdx].sentences[i].tokens[j], tokens[j]))
						continue;

					string curword = feature_word(tokens[j].word);
					string wordnet = feature_wordnet(tokens[j].word, tool);
					string brown = feature_brown(tokens[j].word, tool);
					string bigram = feature_bigram(tokens, j, tool);
					string pos = feature_pos(nlpdocs[docIdx].sentences[i].tokens[j], tool);
					string sst = feature_sst(nlpdocs[docIdx].sentences[i].tokens[j]);

					word_stat[curword]++;
					wordnet_stat[wordnet]++;
					brown_stat[brown]++;
					bigram_stat[bigram]++;
					pos_stat[pos]++;
					sst_stat[sst]++;
				}

				offset += (sentences[i]).length();
			}


		}

		stat2Alphabet(word_stat, m_wordAlphabet, "word");
		stat2Alphabet(wordnet_stat, m_wordnetAlphabet, "wordnet");
		stat2Alphabet(brown_stat, m_brownAlphabet, "brown");
		stat2Alphabet(bigram_stat, m_bigramAlphabet, "bigram");
		stat2Alphabet(pos_stat, m_posAlphabet, "pos");
		stat2Alphabet(sst_stat, m_sstAlphabet, "sst");
	}

	string feature_word(const string& word) {
		string ret = normalize_to_lowerwithdigit(word);
		return ret;
	}

	string feature_wordnet(const string& word, Tool& tool) {

		string norm = normalize_to_lowerwithdigit(word);
		string lemma;
		for(int i=0;i<tool.wn_pos.size();i++) {
			lemma = tool.wn.morphword(norm, tool.wn_pos[i]);
			vector<synset> synsets = tool.wn.get_synsets(lemma, tool.wn_pos[i]);
			if(!synsets.empty()) {
				stringstream ss;
				ss<<synsets[0].id;
				return ss.str();
			}
		}

		return unknownkey;

	}

	string feature_brown(const string& word, Tool& tool) {
		string brownID = tool.brown.get(fox::toLowercase(word));
		if(!brownID.empty())
			return brownID;
		else
			return unknownkey;
	}

	string feature_bigram(vector<fox::Token>& tokens, int idx, Tool& tool) {
		string bigram;

		if(idx>0) {
			bigram = normalize_to_lowerwithdigit(tokens[idx-1].word+"_"+tokens[idx].word);
		} else {
			bigram = normalize_to_lowerwithdigit(nullkey+"_"+tokens[idx].word);
		}
		return bigram;
	}

	string feature_pos(const Token& token, Tool& tool) {
		return token.pos;
	}

	string feature_sst(const Token& token) {
		int pos = token.sst.find("B-");
		if(pos!=-1) {
			return token.sst.substr(pos+2);
		} else {
			pos = token.sst.find("I-");
			if(pos!=-1) {
				return token.sst.substr(pos+2);
			} else
				return token.sst;
		}


	}

	void randomInitNrmat(NRMat<dtype>& nrmat, Alphabet& alphabet) {
		double* emb = new double[alphabet.size()*m_options.wordEmbSize];
		fox::initArray2((double *)emb, (int)alphabet.size(), m_options.wordEmbSize, 0.0);

		vector<string> known;
		map<string, int> IDs;
		alphabet2vectormap(alphabet, known, IDs);

		fox::randomInitEmb((double*)emb, m_options.wordEmbSize, known, unknownkey,
				IDs, true, m_options.initRange);

		nrmat.resize(alphabet.size(), m_options.wordEmbSize);
		array2NRMat((double*) emb, alphabet.size(), m_options.wordEmbSize, nrmat);

		delete[] emb;
	}

	template<typename xpu>
	void averageUnkownEmb(Alphabet& alphabet, LookupTable<xpu>& table) {

		// unknown cannot be trained, use the average embedding
		int unknownID = alphabet.from_string(unknownkey);
		Tensor<cpu, 2, dtype> temp = NewTensor<cpu>(Shape2(1, m_options.wordEmbSize), d_zero);
		int number = table._nVSize-1;
		table._E[unknownID] = 0.0;
		for(int i=0;i<table._nVSize;i++) {
			if(i==unknownID)
				continue;
			table.GetEmb(i, temp);
			table._E[unknownID] += temp[0]/number;
		}

		FreeSpace(&temp);

	}

	void stat2Alphabet(hash_map<string, int>& stat, Alphabet& alphabet, const string& label) {

		cout << label<<" num: " << stat.size() << endl;
		alphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator feat_iter;
		for (feat_iter = stat.begin(); feat_iter != stat.end(); feat_iter++) {
			// if not fine tune, add all the words; if fine tune, add the words considering wordCutOff
			// in order to train unknown
			if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
			  alphabet.from_string(feat_iter->first);
			}
		}
		cout << "alphabet "<< label<<" num: " << alphabet.size() << endl;
		alphabet.set_fixed_flag(true);

	}


	void featureName2ID(Alphabet& alphabet, const string& featureName, vector<int>& vfeatureID) {
		int id = alphabet.from_string(featureName);
		if(id >=0)
			vfeatureID.push_back(id);
		else
			vfeatureID.push_back(0); // assume unknownID is zero
	}

	void loadSstFile(const string& file, vector<Document>& docs) {
		ifstream ifs;
		ifs.open(file.c_str());

		string line;
		int docIdx = -1;
		int sentenceIdx = -1;
		while(getline(ifs, line)) {
			if(line.find("S-1\t")!=-1 || line.find("S-1 ")!=-1) {
				// a new document
				docIdx++;
				// first sentence
				sentenceIdx = 0;

				vector<string> splitted;
				string temp;
				for(unsigned int i=0;i<line.length();i++) {
					if(line[i]==' ' || line[i]=='\t') {
						if(!temp.empty()) {
							splitted.push_back(temp);
							temp.clear();
						}
					} else {
						temp += line[i];
					}

				}
				if(!temp.empty()) {
					splitted.push_back(temp);
					temp.clear();
				}
				assert(splitted.size()-1 == docs[docIdx].sentences[sentenceIdx].tokens.size());
/*				if(splitted.size()-1 != docs[docIdx].sentences[sentenceIdx].tokens.size()) {
					cout<<line<<endl;
					cout<<docIdx<<endl;
					cout<<sentenceIdx<<endl;
					exit(0);
				}*/
				for(int i=0;i<docs[docIdx].sentences[sentenceIdx].tokens.size();i++) {
					docs[docIdx].sentences[sentenceIdx].tokens[i].sst = splitted[i+1];
				}


			} else if(!line.empty()) {
				// other sentence
				sentenceIdx++;

				vector<string> splitted;
				string temp;
				for(unsigned int i=0;i<line.length();i++) {
					if(line[i]==' ' || line[i]=='\t') {
						if(!temp.empty()) {
							splitted.push_back(temp);
							temp.clear();
						}
					} else {
						temp += line[i];
					}

				}
				if(!temp.empty()) {
					splitted.push_back(temp);
					temp.clear();
				}
				assert(splitted.size()-1 == docs[docIdx].sentences[sentenceIdx].tokens.size());
				for(int i=0;i<docs[docIdx].sentences[sentenceIdx].tokens.size();i++) {
					docs[docIdx].sentences[sentenceIdx].tokens[i].sst = splitted[i+1];
				}
			}
		}


		ifs.close();
	}

	void loadNlpFile(const string& file, vector<Document>& docs) {
		ifstream ifs;
		ifs.open(file.c_str());

		string line;
		Document* current = NULL;
		Sent* curSent = NULL;
		while(getline(ifs, line)) {
			if(line.find("#ID#")!=-1) {
				// delete the last sentence of last doc
				if(current!=NULL && !current->sentences.empty())
					current->sentences.erase(current->sentences.end()-1);
				// new doc
				Document doc;
				vector<string> splitted;
				fox::split_bychar(line, splitted, '\t');
				doc.id = splitted[1];
				docs.push_back(doc);
				current = &docs[docs.size()-1];
				Sent sent;
				current->sentences.push_back(sent);
				curSent = &current->sentences[0];
			} else if(line.empty()){
				// new line
				Sent sent;
				current->sentences.push_back(sent);
				curSent = &current->sentences[current->sentences.size()-1];
			} else {
				vector<string> splitted;
				fox::split_bychar(line, splitted, '\t');
				Token token;
				token.word = splitted[0];
				token.begin = atoi(splitted[1].c_str());
				token.end = atoi(splitted[2].c_str());
				token.pos = splitted[3];
				curSent->tokens.push_back(token);
			}



		}

		ifs.close();
	}

	bool isTokenSatisfied(const Token& nlpToken, const Token& rawToken) {
		assert(nlpToken.word==rawToken.word);
		assert(nlpToken.begin==rawToken.begin);
		assert(nlpToken.end==rawToken.end);

		//EnglishPosType type = EnglishPos::getType(nlpToken.pos);
		//if(type != fox::OTHER)
		//if(type == fox::VERB || type == fox::ADV || type==fox::ADJ || type==fox::PREP)
		//if(!Punctuation::isEnglishPunc(rawToken.word[0]))
			return true;
		/*else
			return false;*/
	}

};



#endif /* NNCDR_H_ */
