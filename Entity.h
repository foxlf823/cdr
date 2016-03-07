/*
 * Entity.h
 *
 *  Created on: Dec 20, 2015
 *      Author: fox
 */

#ifndef ENTITY_H_
#define ENTITY_H_
#include <string>

using namespace std;

class Entity {
public:
	Entity() {
		type = "";
		begin = -1;
		end = -1;
		text = "";
		mesh = "-1";
		compositeRole = "";
		begin2 = -1;
		end2 = -1;
	}
/*	virtual ~Entity() {

	}*/

	string type;
	int begin;
	int end; // begin+length
	string text;
	string mesh;
	string compositeRole;
	// there may be some non-continuous entities which have two spans
	// but no more than two spans from the statistics of the training and dev set
	int begin2;
	int end2;
};



#endif /* ENTITY_H_ */
