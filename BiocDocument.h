/*
 * BiocDocument.h
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#ifndef BIOCDOCUMENT_H_
#define BIOCDOCUMENT_H_

#include <string>
#include "Entity.h"
#include "Relation.h"

using namespace std;

class BiocDocument {
public:
	BiocDocument() {

	}
/*	virtual ~BiocDocument() {

	}*/

	string id;
	string title;
	string abstract;
	vector<Entity> entities;
	vector<Relation> relations;
};

#endif /* BIOCDOCUMENT_H_ */
