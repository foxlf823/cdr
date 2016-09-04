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
#include "Token.h"
#include "Sent.h"
#include <sstream>

#include "Classifier_attentionentity.h"
#include "N3L.h"
//#include "wnb/core/wordnet.hh"
#include "Document.h"
#include "EnglishPos.h"
#include "Punctuation.h"
#include "WordNet.h"
#include "Classifier_pooling.h"
#include "Classifier_pooling_entity.h"
#include "Classifier_discrete_neural.h"
#include <set>

using namespace nr;
using namespace std;
//using namespace wnb;

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

	Alphabet m_sparseAlphabet;

	string nullkey;
	string unknownkey;
	string chemicalkey;
	string diseasekey;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
//  Classifier<cpu> m_classifier;
//  Classifier_pooling<cpu> m_classifier;
//  Classifier_pooling_entity<cpu> m_classifier;
  Classifier_discrete_neural<cpu> m_classifier;
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



	void train(const string& trainFile, const string& devFile, const string& testFile, const string& otherDir,
			Tool& tool, bool usedev, const string& predictTestFile,
			const string& trainNlpFile, const string& devNlpFile, const string& testNlpFile) {


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


		vector<Document> trainNlpdocs;
		if(!trainNlpFile.empty()) {
			loadNlpFile(trainNlpFile, trainNlpdocs);
		}

		vector<Document> devNlpdocs;
		if(!devNlpFile.empty()) {
			loadNlpFile(devNlpFile, devNlpdocs);
		}

		vector<Document> testNlpdocs;
		if(!testNlpFile.empty()) {
			loadNlpFile(testNlpFile, testNlpdocs);
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

		m_sparseAlphabet.clear();
		m_sparseAlphabet.from_string(unknownkey);
		m_sparseAlphabet.from_string(nullkey);

		createAlphabet(trainDocuments, tool, trainNlpdocs, true);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devDocuments.empty())
				createAlphabet(devDocuments, tool, devNlpdocs, false);
			if(!testDocuments.empty())
				createAlphabet(testDocuments, tool, testNlpdocs, false);
		}

		NRMat<dtype> wordEmb;
		if(m_options.wordEmbFineTune) {
			if(m_options.embFile.empty()) {
				cout<<"random emb"<<endl;

				randomInitNrmat(wordEmb, m_wordAlphabet);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, true, true);
				// format the words of pre-trained embeddings
				//formatWords(tool.w2v);
				double* emb = new double[m_wordAlphabet.size()*m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);

				tool.w2v->getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);

				delete[] emb;
			}
		} else {
			if(m_options.embFile.empty()) {
				assert(0);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, true, true);
				// format the words of pre-trained embeddings
				//formatWords(tool.w2v);
				double* emb = new double[m_wordAlphabet.size()*m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);

				tool.w2v->getEmbedding((double*)emb, m_options.wordEmbSize, known, unknownkey, IDs);

				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);

				delete[] emb;
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


		m_sparseAlphabet.set_fixed_flag(false);
		vector<Example> trainExamples;
		initialExamples(tool, trainDocuments, trainExamples, trainNlpdocs, true);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;
		m_sparseAlphabet.set_fixed_flag(true);
		cout<<"sparse feature size: "<<m_sparseAlphabet.size()<<endl;




		m_classifier.init(2, wordEmb, wordnetEmb,brownEmb,
				bigramEmb, posEmb, sstEmb, m_sparseAlphabet.size());


		int inputSize = trainExamples.size();
		int batchBlock = inputSize / m_options.batchSize;
		if (inputSize % m_options.batchSize != 0)
			batchBlock++;

		std::vector<int> indexes;
		for (int i = 0; i < inputSize; ++i)
			indexes.push_back(i);

		static Metric eval, metric_dev;
		static vector<Example> subExamples;


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
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words);

					if((m_options.channelMode & 2) == 2) {
						averageUnkownEmb(m_wordnetAlphabet, m_classifier._wordnet);
					}
					if((m_options.channelMode & 4) == 4) {
						averageUnkownEmb(m_brownAlphabet, m_classifier._brown);
					}
					if((m_options.channelMode & 8) == 8) {
						averageUnkownEmb(m_bigramAlphabet, m_classifier._bigram);
					}
					if((m_options.channelMode & 16) == 16) {
						averageUnkownEmb(m_posAlphabet, m_classifier._pos);
					}
					if((m_options.channelMode & 32) == 32) {
						averageUnkownEmb(m_sstAlphabet, m_classifier._sst);
					}
		    	}

		    	metric_dev.reset();
		    	if(usedev) {
		    		toBeOutput.clear();
		    		for(int i=0;i<devDocuments.size();i++) {
		    			vector<Example> tempExamples;
		    			vector<BiocDocument> temp;
		    			temp.push_back(devDocuments[i]);
		    			vector<Document> tempNlp;
		    			tempNlp.push_back(devNlpdocs[i]);
		    			initialExamples(tool, temp, tempExamples, tempNlp, false);


		    			for (int idx = 0; idx < tempExamples.size(); idx++) {
		    				vector<double> scores(2);
		    				m_classifier.predict(tempExamples[idx], scores);
		    				bool predicted = scores[0]>scores[1] ? true:false;

		    				if(predicted)
		    					toBeOutput.push_back(tempExamples[idx]);
		    			}

		    		}
		    		outputToPubtator(toBeOutput, m_options.output);


		    	} else {
		    		for(int i=0;i<devDocuments.size();i++) {
						vector<Example> tempExamples;
						vector<BiocDocument> temp;
						temp.push_back(devDocuments[i]);
						vector<Document> tempNlp;
						tempNlp.push_back(devNlpdocs[i]);
						initialExamples(tool, temp, tempExamples, tempNlp, false);

						metric_dev.overall_label_count += devDocuments[i].relations.size();
						set<string> goldSet;
						for(int k=0;k<devDocuments[i].relations.size();k++)
							goldSet.insert(devDocuments[i].relations[k].chemcalMesh+"_"+devDocuments[i].relations[k].diseaseMesh);

						// example is generated in mention-level, so we get rid of overlapped
						set<string> predictSet;
						for (int idx = 0; idx < tempExamples.size(); idx++) {
							vector<double> scores(2);
							m_classifier.predict(tempExamples[idx], scores);
							bool predicted = scores[0]>scores[1] ? true:false;

							if(predicted) {
								outputToSet(tempExamples[idx], predictSet);
							}
						}

						metric_dev.predicated_label_count += predictSet.size();

						set<string>::iterator predictIt;
						for(predictIt = predictSet.begin();predictIt != predictSet.end();predictIt++) {
							set<string>::iterator goldIt = goldSet.find(*predictIt);
							if(goldIt != goldSet.end())
								metric_dev.correct_label_count++;
						}
		    		}

			    	metric_dev.print();

			    	if (metric_dev.getAccuracy() > best) {
			    		cout << "Exceeds best performance of " << best << endl;
			    		best = metric_dev.getAccuracy();
			    		// if the current exceeds the best, we do the blind test on the test set
			    		// but don't evaluate and store the results for the official evaluation
						if (!testFile.empty()) {
							toBeOutput.clear();

							for (int i = 0; i < testDocuments.size(); i++) {
								vector<Example> tempExamples;
								vector<BiocDocument> temp;
								temp.push_back(testDocuments[i]);
								vector<Document> tempNlp;
								tempNlp.push_back(testNlpdocs[i]);
								initialExamples(tool, temp, tempExamples, tempNlp, false);

				    			for (int idx = 0; idx < tempExamples.size(); idx++) {
				    				vector<double> scores(2);
				    				m_classifier.predict(tempExamples[idx], scores);
				    				bool predicted = scores[0]>scores[1] ? true:false;

				    				if(predicted)
				    					toBeOutput.push_back(tempExamples[idx]);
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
			 const vector<Document>& nlpdocs, bool bStatistic) {
		int ctPositive = 0;
		int ctNegtive = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {


			for(int sentIdx=0;sentIdx<nlpdocs[docIdx].sentences.size();sentIdx++) {

				// find all the entities in the current sentence
				vector<Entity> Bentity;
				findEntityInSent(nlpdocs[docIdx].sentences[sentIdx].begin, nlpdocs[docIdx].sentences[sentIdx].end, documents[docIdx], Bentity);
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
						findEntityInSent(nlpdocs[docIdx].sentences[i].begin, nlpdocs[docIdx].sentences[i].end, documents[docIdx], Aentity);
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
								for(int k=0;k<nlpdocs[docIdx].sentences[j].tokens.size();k++) {
									if(isTokenBeforeEntity(nlpdocs[docIdx].sentences[j].tokens[k], Aentity[a])) {

										/*int type = isTokenInAnyEntity(sents[j].tokens[k], documents[docIdx]);

										if(0 != type)
											featureName2ID(m_wordAlphabet, type==1?chemicalkey:diseasekey, eg.m_before);
										else*/
											featureName2ID(m_wordAlphabet, feature_word(nlpdocs[docIdx].sentences[j].tokens[k].word), eg.m_before);


										if((m_options.channelMode & 2) == 2) {
											featureName2ID(m_wordnetAlphabet, feature_wordnet(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_before_wordnet);
										}
										if((m_options.channelMode & 4) == 4) {
											featureName2ID(m_brownAlphabet, feature_brown(nlpdocs[docIdx].sentences[j].tokens[k].word, tool), eg.m_before_brown);
										}
										if((m_options.channelMode & 8) == 8) {
											featureName2ID(m_bigramAlphabet, feature_bigram(nlpdocs[docIdx].sentences[j].tokens, k, tool), eg.m_before_bigram);
										}
										if((m_options.channelMode & 16) == 16) {
											featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_before_pos);
										}
										if((m_options.channelMode & 32) == 32) {
											featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_before_sst);
										}



									}
									else if(isTokenAfterEntity(nlpdocs[docIdx].sentences[j].tokens[k], Bentity[b])) {

										/*int type = isTokenInAnyEntity(sents[j].tokens[k], documents[docIdx]);

										if(0 != type)
											featureName2ID(m_wordAlphabet, type==1?chemicalkey:diseasekey, eg.m_after);
										else*/
											featureName2ID(m_wordAlphabet, feature_word(nlpdocs[docIdx].sentences[j].tokens[k].word), eg.m_after);

										if((m_options.channelMode & 2) == 2) {
											featureName2ID(m_wordnetAlphabet, feature_wordnet(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_after_wordnet);
										}
										if((m_options.channelMode & 4) == 4) {
											featureName2ID(m_brownAlphabet, feature_brown(nlpdocs[docIdx].sentences[j].tokens[k].word, tool), eg.m_after_brown);
										}
										if((m_options.channelMode & 8) == 8) {
											featureName2ID(m_bigramAlphabet, feature_bigram(nlpdocs[docIdx].sentences[j].tokens, k, tool), eg.m_after_bigram);
										}
										if((m_options.channelMode & 16) == 16) {
											featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_after_pos);
										}
										if((m_options.channelMode & 32) == 32) {
											featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_after_sst);
										}


									}
									else if(isTokenInEntity(nlpdocs[docIdx].sentences[j].tokens[k], chemical)) {

										featureName2ID(m_wordAlphabet, feature_word(nlpdocs[docIdx].sentences[j].tokens[k].word), eg.m_entityFormer);


										if((m_options.channelMode & 2) == 2) {
											featureName2ID(m_wordnetAlphabet, feature_wordnet(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_entityFormer_wordnet);
										}
										if((m_options.channelMode & 4) == 4) {
											featureName2ID(m_brownAlphabet, feature_brown(nlpdocs[docIdx].sentences[j].tokens[k].word, tool), eg.m_entityFormer_brown);
										}
										if((m_options.channelMode & 8) == 8) {
											featureName2ID(m_bigramAlphabet, feature_bigram(nlpdocs[docIdx].sentences[j].tokens, k, tool), eg.m_entityFormer_bigram);
										}
										if((m_options.channelMode & 16) == 16) {
											featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_entityFormer_pos);
										}
										if((m_options.channelMode & 32) == 32) {
											featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_entityFormer_sst);
										}


									} else if(isTokenInEntity(nlpdocs[docIdx].sentences[j].tokens[k], disease)) {

										featureName2ID(m_wordAlphabet, feature_word(nlpdocs[docIdx].sentences[j].tokens[k].word), eg.m_entityLatter);

										if((m_options.channelMode & 2) == 2) {
											featureName2ID(m_wordnetAlphabet, feature_wordnet(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_entityLatter_wordnet);
										}
										if((m_options.channelMode & 4) == 4) {
											featureName2ID(m_brownAlphabet, feature_brown(nlpdocs[docIdx].sentences[j].tokens[k].word, tool), eg.m_entityLatter_brown);
										}
										if((m_options.channelMode & 8) == 8) {
											featureName2ID(m_bigramAlphabet, feature_bigram(nlpdocs[docIdx].sentences[j].tokens, k, tool), eg.m_entityLatter_bigram);
										}
										if((m_options.channelMode & 16) == 16) {
											featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_entityLatter_pos);
										}
										if((m_options.channelMode & 32) == 32) {
											featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_entityLatter_sst);
										}


									} else if(isTokenBetweenTwoEntities(nlpdocs[docIdx].sentences[j].tokens[k], Aentity[a], Bentity[b])){

										/*int type = isTokenInAnyEntity(sents[j].tokens[k], documents[docIdx]);

										if(0 != type)
											featureName2ID(m_wordAlphabet, type==1?chemicalkey:diseasekey, eg.m_middle);
										else*/
											featureName2ID(m_wordAlphabet, feature_word(nlpdocs[docIdx].sentences[j].tokens[k].word), eg.m_middle);

										if((m_options.channelMode & 2) == 2) {
											featureName2ID(m_wordnetAlphabet, feature_wordnet(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_middle_wordnet);
										}
										if((m_options.channelMode & 4) == 4) {
											featureName2ID(m_brownAlphabet, feature_brown(nlpdocs[docIdx].sentences[j].tokens[k].word, tool), eg.m_middle_brown);
										}
										if((m_options.channelMode & 8) == 8) {
											featureName2ID(m_bigramAlphabet, feature_bigram(nlpdocs[docIdx].sentences[j].tokens, k, tool), eg.m_middle_bigram);
										}
										if((m_options.channelMode & 16) == 16) {
											featureName2ID(m_posAlphabet, feature_pos(nlpdocs[docIdx].sentences[j].tokens[k], tool), eg.m_middle_pos);
										}
										if((m_options.channelMode & 32) == 32) {
											featureName2ID(m_sstAlphabet, feature_sst(nlpdocs[docIdx].sentences[j].tokens[k]), eg.m_middle_sst);
										}


									}

								}
							}



							// for concise, we don't judge channel mode here, but it's ok since
							// classifier will not use unnecessary channel
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

							examples.push_back(eg);

							if(eg.m_labels[0]==1) {
								ctPositive++;
							}
							else {
								ctNegtive++;
							}


						}


					}

				}


			}
		}

		if(bStatistic) {
			cout<<"Positive example: "<<ctPositive<<endl;
			cout<<"Negative example: "<<ctNegtive<<endl;
			cout<<"Proportion +:"<< (ctPositive*1.0)/(ctPositive+ctNegtive)
					<<" , -:"<<(ctNegtive*1.0)/(ctPositive+ctNegtive)<<endl;
		}


	}

	void createAlphabet (const vector<BiocDocument>& documents, Tool& tool,
			const vector<Document>& nlpdocs, bool isTrainSet) {

		hash_map<string, int> word_stat;
		hash_map<string, int> wordnet_stat;
		hash_map<string, int> brown_stat;
		hash_map<string, int> bigram_stat;
		hash_map<string, int> pos_stat;
		hash_map<string, int> sst_stat;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {

			for(int i=0;i<nlpdocs[docIdx].sentences.size();i++) {


				for(int j=0;j<nlpdocs[docIdx].sentences[i].tokens.size();j++) {


					string curword = feature_word(nlpdocs[docIdx].sentences[i].tokens[j].word);
					word_stat[curword]++;

					if(isTrainSet && (m_options.channelMode & 2) == 2) {
						string wordnet = feature_wordnet(nlpdocs[docIdx].sentences[i].tokens[j], tool);
						wordnet_stat[wordnet]++;
					}
					if(isTrainSet && (m_options.channelMode & 4) == 4) {
						string brown = feature_brown(nlpdocs[docIdx].sentences[i].tokens[j].word, tool);
						brown_stat[brown]++;
					}
					if(isTrainSet && (m_options.channelMode & 8) == 8) {
						string bigram = feature_bigram(nlpdocs[docIdx].sentences[i].tokens, j, tool);
						bigram_stat[bigram]++;
					}
					if(isTrainSet && (m_options.channelMode & 16) == 16) {
						string pos = feature_pos(nlpdocs[docIdx].sentences[i].tokens[j], tool);
						pos_stat[pos]++;
					}
					if(isTrainSet && (m_options.channelMode & 32) == 32) {
						string sst = feature_sst(nlpdocs[docIdx].sentences[i].tokens[j]);
						sst_stat[sst]++;
					}




				}


			}


		}

		stat2Alphabet(word_stat, m_wordAlphabet, "word");

		if(isTrainSet && (m_options.channelMode & 2) == 2) {
			stat2Alphabet(wordnet_stat, m_wordnetAlphabet, "wordnet");
		}
		if(isTrainSet && (m_options.channelMode & 4) == 4) {
			stat2Alphabet(brown_stat, m_brownAlphabet, "brown");
		}
		if(isTrainSet && (m_options.channelMode & 8) == 8) {
			stat2Alphabet(bigram_stat, m_bigramAlphabet, "bigram");
		}
		if(isTrainSet && (m_options.channelMode & 16) == 16) {
			stat2Alphabet(pos_stat, m_posAlphabet, "pos");
		}
		if(isTrainSet && (m_options.channelMode & 32) == 32) {
			stat2Alphabet(sst_stat, m_sstAlphabet, "sst");
		}
	}

	string feature_word(const string& word) {
		string ret = normalize_to_lowerwithdigit(word);
		return ret;
	}

	string feature_wordnet(const fox::Token& token, Tool& tool) {

/*		string norm = normalize_to_lowerwithdigit(word);
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

		return unknownkey;*/

		string lemmalow = fox::toLowercase(token.lemma);
		char buffer[64] = {0};
		sprintf(buffer, "%s", lemmalow.c_str());

		int pos = -1;
		fox::EnglishPosType type = fox::EnglishPos::getType(token.pos);
		if(type == fox::FOX_NOUN)
			pos = WNNOUN;
		else if(type == fox::FOX_VERB)
			pos = WNVERB;
		else if(type == fox::FOX_ADJ)
			pos = WNADJ;
		else if(type == fox::FOX_ADV)
			pos = WNADV;

		if(pos != -1) {
			string id = fox::getWnID(buffer, pos, 1);
			if(!id.empty())
				return id;
			else
				return unknownkey;
		} else
			return unknownkey;


	}

	string feature_brown(const string& word, Tool& tool) {
		string brownID = tool.brown.get(fox::toLowercase(word));
		if(!brownID.empty())
			return brownID;
		else
			return unknownkey;
	}

	string feature_bigram(const vector<fox::Token>& tokens, int idx, Tool& tool) {
		string bigram;

		if(idx>0) {
			bigram = normalize_to_lowerwithdigit(tokens[idx-1].word+"_"+tokens[idx].word);
		} else {
			bigram = normalize_to_lowerwithdigit(nullkey+"_"+tokens[idx].word);
		}
		return bigram;
	}

	string feature_pos(const fox::Token& token, Tool& tool) {
		return token.pos;
	}

	string feature_sst(const fox::Token& token) {
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



};



#endif /* NNCDR_H_ */

