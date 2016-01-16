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
	virtual ~BiocDocument() {
		/*for(int i=0;i<entities.size();i++) {
			delete entities[i];
		}
		for(int i=0;i<relations.size();i++) {
			delete relations[i];
		}*/
	}

	string id;
	string title;
	string abstract;
	vector<Entity> entities;
	vector<Relation> relations;
};

#endif /* BIOCDOCUMENT_H_ */
