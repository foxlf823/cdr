/*
 * NNcdr.h
 *
 *  Created on: Dec 27, 2015
 *      Author: fox
 */

#ifndef NNCDR_RNN_H_
#define NNCDR_RNN_H_

#include "Options.h"
#include "Tool.h"
#include "FoxUtil.h"
#include "N3Lhelper.h"
#include "Utf.h"
#include "Token.h"
#include "Sent.h"
#include <sstream>
#include "N3L.h"
//#include "wnb/core/wordnet.hh"
#include "Document.h"
#include "EnglishPos.h"
#include "Punctuation.h"
//#include "WordNet.h"
#include <set>
#include "PoolGRNNClassifier.h"


using namespace nr;
using namespace std;
//using namespace wnb;


class NNcdr_rnn {
public:
	Options m_options;
	Alphabet m_wordAlphabet;
	Alphabet m_positionAlphabet;


	string nullkey;
	string unknownkey;
	string chemicalkey;
	string diseasekey;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
  //PoolExGRNNClassifier1<cpu> m_classifier;
  //PoolExGRNNClassifier2<cpu> m_classifier;
  //PoolRNNClassifier<cpu> m_classifier;
  //PoolGRNNClassifier<cpu> m_classifier;
  PoolGRNNClassifier3<cpu> m_classifier;
#endif

  NNcdr_rnn(const Options &options):m_options(options)/*, m_classifier(options)*/ {
		nullkey = "-#null#-";
		unknownkey = "-#unknown#-";
		chemicalkey = "-chem-";
		diseasekey = "-dise-";
	}




	void train(const string& trainFile, const string& devFile, const string& testFile,
			Tool& tool, bool usedev,
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

		m_positionAlphabet.clear();
		m_positionAlphabet.from_string(unknownkey);
		m_positionAlphabet.from_string(nullkey);

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

				randomInitNrmat(wordEmb, m_wordAlphabet, m_options.wordEmbSize);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, false, true);
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
				cout<<"embFile can't be empty if not finetune"<<endl;
				exit(0);
			} else {
				cout<< "load pre-trained emb"<<endl;
				tool.w2v->loadFromBinFile(m_options.embFile, false, true);
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

		NRMat<dtype> positionEmb;
		randomInitNrmat(positionEmb, m_positionAlphabet, m_options.otherEmbSize);


		vector<Example> trainExamples;
		initialExamples(tool, trainDocuments, trainExamples, trainNlpdocs, true);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;


		  m_classifier.init(wordEmb, m_options);
		  m_classifier.resetRemove(m_options.removePool);
		  m_classifier.setDropValue(m_options.dropProb);
		  m_classifier.setWordEmbFinetune(m_options.wordEmbFineTune);

		  m_classifier._position.initial(positionEmb);
		  m_classifier._position.setEmbFineTune(true);


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
		      m_classifier.updateParams(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);

		    }

		    // an iteration end, begin to evaluate
		    if (devDocuments.size() > 0 && (iter+1)% m_options.evalPerIter ==0) {

		    	if(m_options.wordCutOff == 0) {

		    		averageUnkownEmb(m_wordAlphabet, m_classifier._words, m_options.wordEmbSize);

		    	}

		    	metric_dev.reset();
/*		    	if(usedev) {
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


		    	} else {*/
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
		    	//}






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
					if(Bentity[b].compositeRole=="IndividualMention")
						continue;


					for(int i=windowBegin;i<=sentIdx;i++) {
						vector<Entity> Aentity;
						findEntityInSent(nlpdocs[docIdx].sentences[i].begin, nlpdocs[docIdx].sentences[i].end, documents[docIdx], Aentity);
						for(int a=0;a<Aentity.size();a++) {

							if(Aentity[a].mesh == "-1")
								continue;
							/*if(Aentity[a].compositeRole=="CompositeMention")
								continue;*/
							if(Aentity[a].compositeRole=="IndividualMention")
								continue;
							if(Aentity[a].begin >= Bentity[b].begin)
								continue;
							if(Aentity[a].type == Bentity[b].type)
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
							int tkIdx = 0;

							for(int j=i;j<=sentIdx;j++) {
								for(int k=0;k<nlpdocs[docIdx].sentences[j].tokens.size();k++, tkIdx++) {


									if(isTokenInEntity(nlpdocs[docIdx].sentences[j].tokens[k], Aentity[a])) {

										if(eg.formerTkBegin == -1)
											eg.formerTkBegin = tkIdx;

										if(eg.formerTkEnd == -1)
											eg.formerTkEnd = tkIdx;
										else if(eg.formerTkEnd < tkIdx)
											eg.formerTkEnd = tkIdx;

									} else if(isTokenInEntity(nlpdocs[docIdx].sentences[j].tokens[k], Bentity[b])) {

										if(eg.latterTkBegin == -1)
											eg.latterTkBegin = tkIdx;

										if(eg.latterTkEnd == -1)
											eg.latterTkEnd = tkIdx;
										else if(eg.latterTkEnd < tkIdx)
											eg.latterTkEnd = tkIdx;

									}


									Feature feature;

									featureName2ID(m_wordAlphabet, feature_word(nlpdocs[docIdx].sentences[j].tokens[k]), feature.words);


									eg.m_features.push_back(feature);

								}
							}



							eg.docID = documents[docIdx].id;
							if(Aentity[a].type == "Chemical") {
								eg.chemcalMesh = Aentity[a].mesh;
								eg.diseaseMesh = Bentity[b].mesh;

							} else {
								eg.chemcalMesh = Bentity[b].mesh;
								eg.diseaseMesh = Aentity[a].mesh;
							}


/*							assert(eg.formerTkBegin!=-1 && eg.formerTkEnd!=-1);
							assert(eg.latterTkBegin!=-1 && eg.latterTkEnd!=-1);*/
							if(eg.formerTkBegin==-1 && eg.formerTkEnd==-1) {
								cout<<"warning tkIdx=-1: "<<documents[docIdx].id<<" "<<Aentity[a].text<<" "<<Aentity[a].begin<<" "<<Aentity[a].end<<endl;
								continue;
							}
							if(eg.latterTkBegin==-1 && eg.latterTkEnd==-1) {
								cout<<"warning tkIdx=-1: "<<documents[docIdx].id<<" "<<Bentity[b].text<<" "<<Bentity[b].begin<<" "<<Bentity[b].end<<endl;
								continue;
							}

							int seq_size = eg.m_features.size();
						      for (int idx = 0; idx < seq_size; idx++) {
						        Feature& feature = eg.m_features[idx];

						        if (idx < eg.formerTkBegin) {
						        	feature.position1 = featureName2ID(m_positionAlphabet, feature_position(idx-eg.formerTkBegin));
						        	feature.position2 = featureName2ID(m_positionAlphabet, feature_position(idx-eg.latterTkBegin));

						        } else if(idx >= eg.formerTkBegin && idx <= eg.formerTkEnd) {
						        	feature.position1 = featureName2ID(m_positionAlphabet, feature_position(0));
						        	feature.position2 = featureName2ID(m_positionAlphabet, feature_position(idx-eg.latterTkBegin));

								} else if (idx >= eg.latterTkBegin && idx <= eg.latterTkEnd) {
									feature.position1 = featureName2ID(m_positionAlphabet, feature_position(idx-eg.formerTkEnd));
									feature.position2 = featureName2ID(m_positionAlphabet, feature_position(0));

						        } else if (idx > eg.latterTkEnd) {
						        	feature.position1 = featureName2ID(m_positionAlphabet, feature_position(idx-eg.formerTkEnd));
						        	feature.position2 = featureName2ID(m_positionAlphabet, feature_position(idx-eg.latterTkEnd));

						        } else {
						        	feature.position1 = featureName2ID(m_positionAlphabet, feature_position(idx-eg.formerTkEnd));
						        	feature.position2 = featureName2ID(m_positionAlphabet, feature_position(idx-eg.latterTkBegin));

						        }
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

		// position feature is directly generated once.
		if(isTrainSet) {
			hash_map<string, int> position_stat;
			for(int i=0;i<200;i++) {
				stringstream ss1;
				ss1<<i;
				position_stat[ss1.str()] = 10; // in case to be cutoff
				stringstream ss2;
				ss2<<-i;
				position_stat[ss2.str()] = 10;
			}
			stat2Alphabet(position_stat, m_positionAlphabet, "position");
		}

		for(int docIdx=0;docIdx<documents.size();docIdx++) {

			for(int i=0;i<nlpdocs[docIdx].sentences.size();i++) {


				for(int j=0;j<nlpdocs[docIdx].sentences[i].tokens.size();j++) {


					string curword = feature_word(nlpdocs[docIdx].sentences[i].tokens[j].word);
					word_stat[curword]++;

				}


			}


		}

		stat2Alphabet(word_stat, m_wordAlphabet, "word");

	}

	string feature_word(const string& word) {
		string ret = normalize_to_lowerwithdigit(word);
		return ret;
	}

	string feature_word(const fox::Token& token) {
		string ret = normalize_to_lowerwithdigit(token.word);
		return ret;
	}

	string feature_position(int position) {
		stringstream ss;
		ss<<position;
		return ss.str();
	}


	void randomInitNrmat(NRMat<dtype>& nrmat, Alphabet& alphabet, int embSize) {
		double* emb = new double[alphabet.size()*embSize];
		fox::initArray2((double *)emb, (int)alphabet.size(), embSize, 0.0);

		vector<string> known;
		map<string, int> IDs;
		alphabet2vectormap(alphabet, known, IDs);

		fox::randomInitEmb((double*)emb, embSize, known, unknownkey,
				IDs, false, m_options.initRange);

		nrmat.resize(alphabet.size(), embSize);
		array2NRMat((double*) emb, alphabet.size(), embSize, nrmat);

		delete[] emb;
	}

	template<typename xpu>
	void averageUnkownEmb(Alphabet& alphabet, LookupTable<xpu>& table, int embSize) {

		// unknown cannot be trained, use the average embedding
		int unknownID = alphabet.from_string(unknownkey);
		Tensor<cpu, 2, dtype> temp = NewTensor<cpu>(Shape2(1, embSize), d_zero);
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

	int featureName2ID(Alphabet& alphabet, const string& featureName) {
		int id = alphabet.from_string(featureName);
		if(id >=0)
			return id;
		else
			return 0; // assume unknownID is zero
	}


};



#endif /* NNCDR_H_ */

