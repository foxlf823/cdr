/*
 * Example.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_EXAMPLE_H_
#define SRC_EXAMPLE_H_


using namespace std;

class Example {

public:
  vector<int> m_labels;

  vector<int> m_before;
  vector<int> m_entityFormer;
  vector<int> m_entityLatter;
  vector<int> m_middle;
  vector<int> m_after;

  vector<int> m_before_wordnet;
  vector<int> m_middle_wordnet;
  vector<int> m_after_wordnet;
  vector<int> m_entityFormer_wordnet;
  vector<int> m_entityLatter_wordnet;

  vector<int> m_before_brown;
  vector<int> m_middle_brown;
  vector<int> m_after_brown;
  vector<int> m_entityFormer_brown;
  vector<int> m_entityLatter_brown;

  vector<int> m_before_bigram;
  vector<int> m_middle_bigram;
  vector<int> m_after_bigram;
  vector<int> m_entityFormer_bigram;
  vector<int> m_entityLatter_bigram;

  vector<int> m_before_pos;
  vector<int> m_middle_pos;
  vector<int> m_after_pos;
  vector<int> m_entityFormer_pos;
  vector<int> m_entityLatter_pos;

  vector<int> m_before_sst;
  vector<int> m_middle_sst;
  vector<int> m_after_sst;
  vector<int> m_entityFormer_sst;
  vector<int> m_entityLatter_sst;


// for evaluate
	string chemcalMesh;
	string diseaseMesh;
  string docID;
public:
  Example()
  {

  }
/*  virtual ~Example()
  {

  }*/



};

#endif /* SRC_EXAMPLE_H_ */
