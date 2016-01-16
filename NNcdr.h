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
#include "ClassifierWithoutEntity.h"

using namespace nr;
using namespace std;

// a nn model to classify the cdr relation
class NNcdr {
public:
	Options m_options;
	Alphabet m_wordAlphabet;
	string nullkey;
	string unknownkey;
	string chemicalkey;
	string diseasekey;

#if USE_CUDA==1
  Classifier<gpu> m_classifier;
#else
  Classifier<cpu> m_classifier;
//  ClassifierWithoutEntity<cpu> m_classifier;
#endif

	NNcdr(const Options &options):m_options(options), m_classifier(options) {
		nullkey = "-null-";
		unknownkey = "-unknown-";
		chemicalkey = "-chem-";
		diseasekey = "-dise-";
	}
	virtual ~NNcdr() {

	}

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

	void train(const string& trainFile, const string& devFile, const string& testFile,
			Tool& tool) {

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

		cout << "Creating Alphabet..." << endl;
		m_wordAlphabet.clear();
		m_wordAlphabet.from_string(unknownkey);
		m_wordAlphabet.from_string(nullkey);
		m_wordAlphabet.from_string(chemicalkey);
		m_wordAlphabet.from_string(diseasekey);
		createAlphabet(trainDocuments, tool);

		if (!m_options.wordEmbFineTune) {
			// if not fine tune, use all the data to build alphabet
			if(!devDocuments.empty())
				createAlphabet(devDocuments, tool);
			if(!testDocuments.empty())
				createAlphabet(testDocuments, tool);
		}

		NRMat<dtype> wordEmb;
		if(m_options.wordEmbFineTune) {
			if(m_options.embFile.empty()) {
				cout<<"random emb"<<endl;

				double emb[m_wordAlphabet.size()][m_options.wordEmbSize];
				fox::initArray2((double *)emb, (int)m_wordAlphabet.size(), m_options.wordEmbSize, 0.0);
				vector<string> known;
				map<string, int> IDs;
				alphabet2vectormap(m_wordAlphabet, known, IDs);


				fox::randomInitEmb((double*)emb, m_options.wordEmbSize, known, unknownkey,
						IDs, true, m_options.initRange);


				wordEmb.resize(m_wordAlphabet.size(), m_options.wordEmbSize);
				array2NRMat((double*) emb, m_wordAlphabet.size(), m_options.wordEmbSize, wordEmb);

				/*int id = m_wordAlphabet.from_string(unknownkey);
				for(int j=0;j<wordEmb.ncols();j++) {
											cout<<wordEmb[id][j]<<" ";
										}
										cout<<endl;*/
					/*for(int i=0;i<wordEmb.nrows();i++) {
						for(int j=0;j<wordEmb.ncols();j++) {
							cout<<wordEmb[i][j]<<" ";
						}
						cout<<endl;
					}*/
				//fox::releaseVector(known);

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


		m_classifier.init(wordEmb);

		vector<Example> trainExamples;
		initialExamples(tool, trainDocuments, trainExamples,false);
		cout<<"Total train example number: "<<trainExamples.size()<<endl;
		vector<Example> devExamples;
		if(!devFile.empty()) {
			initialExamples(tool, devDocuments, devExamples,false);
			cout<<"Total dev example number: "<<devExamples.size()<<endl;
		}
		vector<Example> testExamples;
		if(!testFile.empty()) {
			initialExamples(tool, testDocuments, testExamples,false);
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
		    		// unknown cannot be trained, use the average embedding
		    		int unknownID = m_wordAlphabet.from_string(unknownkey);
		    		Tensor<cpu, 2, dtype> temp = NewTensor<cpu>(Shape2(1, m_options.wordEmbSize), d_zero);
		    		int number = m_classifier._words._nVSize-1;
		    		m_classifier._words._E[unknownID] = 0.0;
		    		for(int i=0;i<m_classifier._words._nVSize;i++) {
		    			if(i==unknownID)
		    				continue;
		    			m_classifier._words.GetEmb(i, temp);
		    			m_classifier._words._E[unknownID] += temp[0]/number;
		    		}

		    		FreeSpace(&temp);
		    	}

		    	//cout<<"begin evaluate"<<endl;
		    	metric_dev.reset();
		    	for (int idx = 0; idx < devExamples.size(); idx++) {
		    		/*cout<<devExamples[idx].docID<<" "
		    				<<devExamples[idx].chemcalMesh<<" "<<devExamples[idx].diseaseMesh<<endl;*/
		    		bool predicted = predict(devExamples[idx]);

					evaluate(devExamples[idx], predicted, metric_dev);
		    	}
		    	//cout<<"end evaluate"<<endl;
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


		    } // devExamples > 0

		} // for iter




		m_classifier.release();

	}

	void initialExamples(Tool& tool, const vector<BiocDocument>& documents, vector<Example>& examples,
			bool bStatistics) {
		int unknownID = m_wordAlphabet.from_string(unknownkey);
		int ctPositive = 0;
		int ctNegtive = 0;

		for(int docIdx=0;docIdx<documents.size();docIdx++) {
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
								ctPositive++;
							} else {
								// negative
								eg.m_labels.push_back(0);
								eg.m_labels.push_back(1);
								ctNegtive++;
							}



							for(int j=i;j<=sentIdx;j++) {
								for(int k=0;k<sents[j].tokens.size();k++) {
									if(isTokenBeforeEntity(sents[j].tokens[k], Aentity[a])) {
										string curword = normalize_to_lowerwithdigit(sents[j].tokens[k].word);
										/*int type = isTokenInAnyEntity(sents[j].tokens[k], documents[docIdx]);
										if(type!=0)
											curword = type==1?chemicalkey:diseasekey;*/

										int id = m_wordAlphabet.from_string(curword);
										if(id >=0)
											eg.m_before.push_back(id);
										else
											eg.m_before.push_back(unknownID);
									}
									else if(isTokenAfterEntity(sents[j].tokens[k], Bentity[b])) {
										string curword = normalize_to_lowerwithdigit(sents[j].tokens[k].word);
										/*int type = isTokenInAnyEntity(sents[j].tokens[k], documents[docIdx]);
										if(type!=0)
											curword = type==1?chemicalkey:diseasekey;*/


										int id = m_wordAlphabet.from_string(curword);
										if(id >=0)
											eg.m_after.push_back(id);
										else
											eg.m_after.push_back(unknownID);
									}
									else if(isTokenInEntity(sents[j].tokens[k], Aentity[a])) {
										string curword = normalize_to_lowerwithdigit(sents[j].tokens[k].word);
										int id = m_wordAlphabet.from_string(curword);
										if(id >=0)
											eg.m_entityFormer.push_back(id);
										else
											eg.m_entityFormer.push_back(unknownID);
									} else if(isTokenInEntity(sents[j].tokens[k], Bentity[b])) {
										string curword = normalize_to_lowerwithdigit(sents[j].tokens[k].word);
										int id = m_wordAlphabet.from_string(curword);
										if(id >=0)
											eg.m_entityLatter.push_back(id);
										else
											eg.m_entityLatter.push_back(unknownID);
									} else if(isTokenBetweenTwoEntities(sents[j].tokens[k], Aentity[a], Bentity[b])){
										string curword = normalize_to_lowerwithdigit(sents[j].tokens[k].word);
										/*int type = isTokenInAnyEntity(sents[j].tokens[k], documents[docIdx]);
										if(type!=0)
											curword = type==1?chemicalkey:diseasekey;*/


										int id = m_wordAlphabet.from_string(curword);
										if(id >=0)
											eg.m_middle.push_back(id);
										else
											eg.m_middle.push_back(unknownID);

									}

								}
							}

							// in case that null
							if(eg.m_before.size()==0) {
								eg.m_before.push_back(m_wordAlphabet.from_string(nullkey));
							}
							if(eg.m_entityFormer.size()==0) {
								eg.m_entityFormer.push_back(m_wordAlphabet.from_string(nullkey));
							}
							if(eg.m_entityLatter.size()==0) {
								eg.m_entityLatter.push_back(m_wordAlphabet.from_string(nullkey));
							}
							if(eg.m_middle.size()==0) {
								eg.m_middle.push_back(m_wordAlphabet.from_string(nullkey));
							}
							if(eg.m_after.size()==0) {
								eg.m_after.push_back(m_wordAlphabet.from_string(nullkey));
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

	void createAlphabet (const vector<BiocDocument>& documents, Tool& tool) {

		hash_map<string, int> word_stat;

		for(int i=0;i<documents.size();i++) {
			//string text = documents[i]->title+" "+documents[i]->abstract;
			string text = documents[i].title+" "+documents[i].abstract;
			vector<string> sentences;
			tool.sentSplitter.splitWithFilters(text, sentences);

			int offset = 0;
			for(int i=0;i<sentences.size();i++) {
				vector<fox::Token> tokens;
				tool.tokenizer.tokenize(offset, sentences[i],tokens);

				for(int j=0;j<tokens.size();j++) {
					string curword = normalize_to_lowerwithdigit(tokens[j].word);
					word_stat[curword]++;
				}

				//fox::releaseVector(tokens);

				offset += (sentences[i]).length();
			}


			//fox::releaseVector(sentences);
		}

		cout << "word num: " << word_stat.size() << endl;

		m_wordAlphabet.set_fixed_flag(false);
		hash_map<string, int>::iterator feat_iter;
		for (feat_iter = word_stat.begin(); feat_iter != word_stat.end(); feat_iter++) {
			// if not fine tune, add all the words; if fine tune, add the words considering wordCutOff
			// in order to train unknown
			if (!m_options.wordEmbFineTune || feat_iter->second > m_options.wordCutOff) {
			  m_wordAlphabet.from_string(feat_iter->first);
			}
		}

		cout << "alphabet words num: " << m_wordAlphabet.size() << endl;

		m_wordAlphabet.set_fixed_flag(true);

	}
};



#endif /* NNCDR_H_ */
