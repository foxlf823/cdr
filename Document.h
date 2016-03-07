/*
 * Document.h
 *
 *  Created on: Feb 29, 2016
 *      Author: fox
 */

#ifndef DOCUMENT_H_
#define DOCUMENT_H_
#include "Sent.h"

using namespace fox;

class Document {
public:
	Document() {

	}
/*	virtual ~Document() {

	}*/

	string id;
	vector<Sent> sentences;
};



#endif /* DOCUMENT_H_ */
