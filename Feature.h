
#ifndef SRC_FEATURE_H_
#define SRC_FEATURE_H_

#include <vector>

using namespace std;
class Feature {

public:
	vector<int> words;
	int pos;
	int sst;
	int ner;
	int position1;
	int position2;

public:
	Feature() {
	}
	virtual ~Feature() {

	}

	void clear() {
		words.clear();

	}
};

#endif /* SRC_FEATURE_H_ */
