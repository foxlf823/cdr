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
// for evaluate
	string chemcalMesh;
	string diseaseMesh;
  string docID;
public:
  Example()
  {

  }
  virtual ~Example()
  {

  }



};

#endif /* SRC_EXAMPLE_H_ */
