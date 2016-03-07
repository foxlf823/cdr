/*
 * utils.h
 *
 *  Created on: Dec 20, 2015
 *      Author: fox
 */

#ifndef UTILS_H_
#define UTILS_H_

/*
 * cdr.cpp
 *
 *  Created on: Dec 19, 2015
 *      Author: fox
 */

#include <stdio.h>
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <vector>
#include "BiocDocument.h"
#include "Word2Vec.h"
#include "Utf.h"
#include "Entity.h"
#include "Token.h"
#include "Example.h"
#include "FoxUtil.h"
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

int offsetcount = 0;

void outputToPubtator(const vector<Example>& examples, const string& path) {
	ofstream m_outf;
	m_outf.open(path.c_str());

	for(int i=0;i<examples.size();i++) {
		vector<Example> temps1;
		if(-1 != examples[i].chemcalMesh.find("|")) {
			vector<string> vec;
			fox::split_bychar(examples[i].chemcalMesh, vec, '|');
			for(int j=0;j<vec.size();j++) {
				Example ch = examples[i];
				ch.chemcalMesh = vec[j];
				temps1.push_back(ch);
			}
		} else {
			temps1.push_back(examples[i]);
		}

		vector<Example> temps2;
		for(int k=0;k<temps1.size();k++) {
			if(-1 != temps1[k].diseaseMesh.find("|")) {
				vector<string> vec;
				fox::split_bychar(temps1[k].diseaseMesh, vec, '|');
				for(int j=0;j<vec.size();j++) {
					Example ch = temps1[k];
					ch.diseaseMesh = vec[j];
					temps2.push_back(ch);
				}
			} else {
				temps2.push_back(temps1[k]);
			}
		}


		for(int k=0;k<temps2.size();k++) {
			m_outf << temps2[k].docID << "\t"<< "CID"<<"\t"<<temps2[k].chemcalMesh<<
				"\t"<<temps2[k].diseaseMesh<<endl;
		}
	}
	m_outf.close();
}

bool isTokenBeforeEntity(const fox::Token& tok, const Entity& entity) {
	if(tok.begin<entity.begin)
		return true;
	else
		return false;
}

bool isTokenAfterEntity(const fox::Token& tok, const Entity& entity) {
	if(entity.compositeRole=="IndividualMention") {
		if(entity.begin2==-1 && entity.end2==-1) {
			if(tok.end>entity.end)
				return true;
			else
				return false;
		} else {
			if(tok.end>entity.end2)
				return true;
			else
				return false;
		}

	} else {
		if(tok.end>entity.end)
			return true;
		else
			return false;
	}

}

bool isTokenInEntity(const fox::Token& tok, const Entity& entity) {
	if(entity.compositeRole=="IndividualMention") {
		if(entity.begin2==-1 && entity.end2==-1) {
			if(tok.begin>=entity.begin && tok.end<=entity.end)
				return true;
			else
				return false;
		} else {
			if((tok.begin>=entity.begin && tok.end<=entity.end) ||
					(tok.begin>=entity.begin2 && tok.end<=entity.end2))
				return true;
			else
				return false;
		}

	} else {
		if(tok.begin>=entity.begin && tok.end<=entity.end)
			return true;
		else
			return false;
	}

}

// 0-not in, 1-chemical, 2-disease
int isTokenInAnyEntity(const fox::Token& tok, const BiocDocument& doc) {
	for(int i=0;i<doc.entities.size();i++) {
		const Entity& entity = doc.entities[i];

		if(entity.compositeRole=="IndividualMention") {
			if(entity.begin2==-1 && entity.end2==-1) {
				if(tok.begin>=entity.begin && tok.end<=entity.end)
					return entity.type=="Chemical" ? 1:2;
			} else {
				if((tok.begin>=entity.begin && tok.end<=entity.end) ||
						(tok.begin>=entity.begin2 && tok.end<=entity.end2))
					return entity.type=="Chemical" ? 1:2;
			}

		} else {
			if(tok.begin>=entity.begin && tok.end<=entity.end)
				return entity.type=="Chemical" ? 1:2;
		}
	}

	return 0;
}

bool isTokenBetweenTwoEntities(const fox::Token& tok, const Entity& former, const Entity& latter) {
	if(former.compositeRole=="IndividualMention") {
		if(former.end2 != -1) {
			if(tok.begin>=former.end2 && tok.end<=latter.begin)
				return true;
			else
				return false;
		} else {
			if(tok.begin>=former.end && tok.end<=latter.begin)
				return true;
			else
				return false;
		}

	} else {
		if(tok.begin>=former.end && tok.end<=latter.begin)
			return true;
		else
			return false;
	}
}

// sentence spans from begin(include) to end(exclude), sorted because doc.entities are sorted
void findEntityInSent(int begin, int end, const BiocDocument& doc, vector<Entity>& results) {
	for(int i=0;i<doc.entities.size();i++) {
		if(doc.entities[i].begin >= begin && doc.entities[i].end <= end)
			results.push_back(doc.entities[i]);
	}
}

// whether or not two entities has an ADE relation, judge by mesh ID
bool isADE(const Entity& a, const Entity& b, const BiocDocument& doc) {
	vector<string> a_mesh;
	fox::split_bychar(a.mesh, a_mesh, '|');

	vector<string> b_mesh;
	fox::split_bychar(b.mesh, b_mesh, '|');

	for(int i=0;i<a_mesh.size();i++) {
		for(int j=0;j<b_mesh.size();j++) {
			for(int k=0;k<doc.relations.size();k++) {
				if((doc.relations[k].chemcalMesh==a_mesh[i] && doc.relations[k].diseaseMesh==b_mesh[j]) ||
					(doc.relations[k].chemcalMesh==b_mesh[j] && doc.relations[k].diseaseMesh==a_mesh[i])) {
					return true;
				}
			}
		}
	}

	return false;
}

void parseNode(xmlNodePtr node, xmlNodePtr parent, xmlDocPtr doc, vector<BiocDocument>& documents)
{
    if (!xmlStrcmp(node->name, (const xmlChar *)"document")) {
    	BiocDocument biocDoc;
    	documents.push_back(biocDoc);
    } else if(!xmlStrcmp(node->name, (const xmlChar *)"id") && !xmlStrcmp(parent->name, (const xmlChar *)"document")) {
    	xmlChar *temp = xmlNodeListGetString(doc, node->xmlChildrenNode, 1);
    	documents.back().id = (char*)temp;
    	xmlFree(temp);

    } else if(!xmlStrcmp(node->name, (const xmlChar *)"passage")) {
    	bool isTitle = false;
    	xmlNodePtr child = node->xmlChildrenNode;
    	while (child != NULL) {
    		if(XML_ELEMENT_NODE == child->type) {
    			if(!xmlStrcmp(child->name, (const xmlChar *)"infon")) {
    				xmlChar *temp = xmlNodeGetContent(child->xmlChildrenNode);
    				if(!xmlStrcmp(temp, (const xmlChar *)"title")) {
    					isTitle = true;
    				} else if(!xmlStrcmp(temp, (const xmlChar *)"abstract")) {
    					isTitle = false;
    				}
    				xmlFree(temp);
    			} else if(!xmlStrcmp(child->name, (const xmlChar *)"text")) {
    				xmlChar *temp = xmlNodeGetContent(child->xmlChildrenNode);
    				if(isTitle)
    					documents.back().title = (char*)temp;
    				else {
    					if(temp!=NULL) // some abstract may be null
    						documents.back().abstract = (char*)temp;
    				}
    				xmlFree(temp);
				} else if(!xmlStrcmp(child->name, (const xmlChar *)"annotation")) {
					//Entity* entity = new Entity();
					Entity entity;
					xmlNodePtr annotationChild = child->xmlChildrenNode;
					offsetcount=0;
					while(NULL != annotationChild) {
						if(XML_ELEMENT_NODE == annotationChild->type) {
							if(!xmlStrcmp(annotationChild->name, (const xmlChar *)"infon")) {
								xmlChar* temp = xmlGetProp(annotationChild,BAD_CAST "key");
								if(!xmlStrcmp(temp, (const xmlChar *)"type")) {
									xmlChar *type = xmlNodeGetContent(annotationChild->xmlChildrenNode);
									entity.type = (char*)type;
									xmlFree(type);
								} else if(!xmlStrcmp(temp, (const xmlChar *)"MESH")) {
									xmlChar *mesh = xmlNodeGetContent(annotationChild->xmlChildrenNode);
									if(mesh!=NULL) // bug of BC5CDR-converter
										entity.mesh = (char*)mesh;
									xmlFree(mesh);
								} else if(!xmlStrcmp(temp, (const xmlChar *)"CompositeRole")) {
									xmlChar *compositeRole = xmlNodeGetContent(annotationChild->xmlChildrenNode);
									entity.compositeRole = (char*)compositeRole;
									xmlFree(compositeRole);
								}
								xmlFree(temp);

							} else if(!xmlStrcmp(annotationChild->name, (const xmlChar *)"location")) {
								xmlChar* offset = xmlGetProp(annotationChild,BAD_CAST "offset");
								xmlChar* length = xmlGetProp(annotationChild,BAD_CAST "length");
								if(offsetcount==0) {
									entity.begin = atoi((char*)offset);
									entity.end = entity.begin+atoi((char*)length);
									offsetcount++;
								} else if(offsetcount==1) {
									entity.begin2 = atoi((char*)offset);
									entity.end2 = entity.begin2+atoi((char*)length);
									offsetcount++;
								}
								xmlFree(offset);
								xmlFree(length);

							} else if(!xmlStrcmp(annotationChild->name, (const xmlChar *)"text")) {
								xmlChar *temp = xmlNodeGetContent(annotationChild->xmlChildrenNode);
								entity.text = (char*)temp;
								xmlFree(temp);
							}
						}
						annotationChild = annotationChild->next;
					}
					documents.back().entities.push_back(entity);
				}
    		}
    		child = child->next;
    	}


    } else if(!xmlStrcmp(node->name, (const xmlChar *)"relation")) {
    	//Relation* relation = new Relation();
    	Relation relation;
    	xmlNodePtr child = child = node->xmlChildrenNode;
    	while(NULL != child) {
    		if(XML_ELEMENT_NODE == child->type) {
    			if(!xmlStrcmp(child->name, (const xmlChar *)"infon")) {
    				xmlChar* prop = xmlGetProp(child,BAD_CAST "key");
    				if(!xmlStrcmp(prop, (const xmlChar *)"Chemical")) {
    					xmlChar* temp = xmlNodeGetContent(child->xmlChildrenNode);
    					relation.chemcalMesh = (char*)temp;
    					xmlFree(temp);
    				} else if(!xmlStrcmp(prop, (const xmlChar *)"Disease")) {
    					xmlChar* temp = xmlNodeGetContent(child->xmlChildrenNode);
						relation.diseaseMesh = (char*)temp;
						xmlFree(temp);
    				}
    				xmlFree(prop);
    			}
    		}
    		child = child->next;
    	}
    	documents.back().relations.push_back(relation);
    }



    xmlNodePtr child = NULL;
    child = node->xmlChildrenNode;
    while (child != NULL) {
    	if(XML_ELEMENT_NODE == child->type) {
			parseNode(child, node, doc, documents);
    	}
    	child = child->next;
    }
}



int parseBioc(const string& xmlFilePath, vector<BiocDocument>& documents)
{
    xmlDocPtr doc;           //定义解析文件指针
    xmlNodePtr rootNode;      //定义结点指针(你需要他为了在各个结点间移动)

    // doc = xmlParseFile(argv[1]);
    doc = xmlReadFile(xmlFilePath.c_str(),"utf-8", XML_PARSE_NOBLANKS);
    rootNode = xmlDocGetRootElement(doc);
    parseNode(rootNode, NULL, doc, documents);

    xmlFreeDoc(doc);
    return 0;

}

int parseBiocDir(const string& xmlDir, vector<BiocDocument>& documents)
{
	struct dirent* ent = NULL;
	DIR *pDir;
	pDir = opendir(xmlDir.c_str());

	while (NULL != (ent = readdir(pDir))) {

		if (ent->d_type == 8) {
			//file
			if(ent->d_name[0]=='.')
				continue;
			//printf("%s\n", ent->d_name);
			xmlDocPtr doc;           //定义解析文件指针
			xmlNodePtr rootNode;      //定义结点指针(你需要他为了在各个结点间移动)
			vector<BiocDocument> temp;
			string path = xmlDir;
			path += "/";
			path += ent->d_name;

			doc = xmlReadFile(path.c_str(),"utf-8", XML_PARSE_NOBLANKS);
			rootNode = xmlDocGetRootElement(doc);
			parseNode(rootNode, NULL, doc, temp);
			for(int i=0;i<temp.size();i++)
				documents.push_back(temp[i]);

			xmlFreeDoc(doc);
		}
	}
	closedir(pDir);


    return 0;
}


// format the words of pre-trained embeddings, if needed
void formatWords(fox::Word2Vec& w2v) {
	map<string, float*> formatted;
	for(map<string, float*>::iterator it = w2v.wordMap.begin();
			it != w2v.wordMap.end(); it++) {
		string str = normalize_to_lowerwithdigit(it->first);
		float * value = it->second;
		formatted.insert(map<string, float*>::value_type(str, value));
	}

	w2v.wordMap = formatted;
}


#endif /* UTILS_H_ */
