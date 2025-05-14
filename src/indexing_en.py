import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
import torch
import numpy as np
import json
import pandas as pd
import weaviate
from sentence_transformers import SentenceTransformer
import unicodedata

with open('/di_ord_occ.json', 'r', encoding='utf-8') as file:
    diz = json.load(file)  



key = 'caq***'
url = "https://***"

# client Weaviate
client = weaviate.Client(
    url=url,  
    auth_client_secret=weaviate.auth.AuthApiKey(api_key=key), 
)


# model to performe vectorization
model_rag = SentenceTransformer('all-mpnet-base-v2', device="cuda")

# function for text vectorization
def generate_embeddings(entry, model=model_rag):
        embedding = model.encode(entry, convert_to_numpy=True)
        return embedding
    
    
    

def remove_unicode(stringa):
    string_normaliz = unicodedata.normalize('NFKD', stringa)
    string_ascii = string_normaliz.encode('ascii', 'ignore').decode('ascii')
    return stringa.replace("\\\\","-").replace("\\","-").replace("/","-").replace("//","-")


vett_text = [remove_unicode(j['text']) for i,j in diz.items()]


vett = generate_embeddings(vett_text)



raw_docs = []
tot=0
for key, value in diz.items():
    ISCO_code=value['ISCO_code']
    ISCO_label=value['ISCO_label']
    ISCO_description=value['ISCO_description']
    occup_code =key
    occupation_label=value['occupation_label']
    occupation_alt_label=value['occupation_alternative_label']
    occupation_description=value['occupation_description']
    text = value['text']
    skills = ', '.join([j['skill_alt_label'] for j in value['skill']['skills']])
    skills_essential = ', '.join([j['skill_alt_label'] for j in value['skill']['skills'] if j['skill_type']=='essential'])
    skills_optional = ', '.join([j['skill_alt_label'] for j in value['skill']['skills'] if j['skill_type']=='optional'])
    knowledge = ','.join([j['skill_alt_label'] for j in value['skill']['knowledge']])
    knowledge_essential = ','.join([j['skill_alt_label'] for j in value['skill']['knowledge'] if j['skill_type']=='essential'])
    knowledge_optional = ','.join([j['skill_alt_label'] for j in value['skill']['knowledge'] if j['skill_type']=='optional'])

    if len(skills)!=0:
        meta = {
                "ISCO_code":remove_unicode(str(ISCO_code)),
                "ISCO_label" : remove_unicode(str(ISCO_label)),
                "ISCO_desc" : remove_unicode(str(ISCO_description)),
                "occup_code" : remove_unicode(str(occup_code)),
                "occup_label" : remove_unicode(str(occupation_label)),
                "occup_alt_label" : remove_unicode(str(occupation_alt_label)),
                "occup_desc" : remove_unicode(str(occupation_description)),
                "skills" : remove_unicode(str(skills)),
                "skills_essential" :remove_unicode(str(skills_essential)),
                "skills_optional" :remove_unicode(str(skills_optional)),
                "knowledge" : remove_unicode(str(knowledge)),
                "knowledge_essential" : remove_unicode(str(knowledge_essential)),
                "knowledge_optional" : remove_unicode(str(knowledge_optional)),
                "text":text,
                }
    else:
        meta = {
                "ISCO_code":remove_unicode(str(ISCO_code)),
                "ISCO_label" : remove_unicode(str(ISCO_label)),
                "ISCO_desc" : remove_unicode(str(ISCO_description)),
                "occup_code" : 'NONE',
                "occup_label" : 'NONE',
                "occup_alt_label" : 'NONE',
                "occup_desc" : 'NONE',
                "skills" : 'NONE',
                "skills_essential" :'NONE',
                "skills_optional" :'NONE',
                "knowledge" : 'NONE',
                "knowledge_essential" : 'NONE',
                "knowledge_optional" : 'NONE',
                "text":text,
                }
    embeddings = vett[tot]
    tot +=1
    raw_docs.append({'text':meta, 'vector': embeddings})






# dict of the documento based on JSONL structure
def schema(d):
    dict ={
                "ISCO_code":d.get("ISCO_code",""),
                "ISCO_label" :d.get("ISCO_label",""),
                "ISCO_desc" :d.get("ISCO_desc",""),
                "occup_code" :d.get("occup_code",""),
                "occup_label" :d.get("occup_label",""),
                "occup_alt_label" :d.get("occup_alt_label",""),
                "occup_desc" :d.get("occup_desc",""),
                "skills" :d.get("skills",""),
                "skills_essential" :d.get("skills_essential",""),
                "skills_optional" :d.get("skills_optional",""),
                "knowledge" :d.get("knowledge",""),
                "knowledge_essential" :d.get("knowledge_essential",""),
                "knowledge_optional" :d.get("knowledge_optional",""),
                "text":d.get("text",""),
                }
    return dict



def class_schema(name_class):
    dict = {
        "class": name_class,
        "vectorizer": "none",
    }
    return dict


# Weaviate class
if client.schema.exists('Desc_occ_en'):
    client.schema.delete_class('Desc_occ_en')
    
    
class_cod = class_schema('Desc_occ_en')
client.schema.create_class(class_cod)



# data load and indicization
print("\n\n Tot: ", len(raw_docs), "\n")


client.batch.configure(batch_size=100)  # Configure batch
with client.batch as batch:
    for i, d in enumerate(raw_docs):
        if (i % 200 == 0):
            print(f"importing JSON block: {i + 1}")
        
        properties = schema(d['text'])
        batch.add_data_object(properties,class_name="Desc_occ_en", vector=d['vector'])
            
            
            
print("\n\nOK!!!!\n\n")