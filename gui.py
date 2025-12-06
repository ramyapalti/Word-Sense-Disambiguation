import numpy as np 
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors

# nltk.download("wordnet")
# nltk.download("stopwords")

model_w2v = KeyedVectors.load("./../model_w2v.wordvectors", mmap='r')

# get embedding of a sentence - sent is set of words
def get_embed(sent):
    emb = []
    len_emb = 0
    for word in sent:
        try:
            emb.append(model_w2v.get_vector(word))
            len_emb += 1
        except:
            continue
    # embedding is avergae of word vectors in sent
    emb = np.sum(emb, axis = 0)/len_emb
    return emb

# compute cosine similarity between sense_emb and context_emb
def compute_score(gloss, context_bag):
    
    # get sense embedding from gloss
    sense_emb = get_embed(gloss)

    #get context embedding
    context_embed = get_embed(context_bag)

    # perform cosine score
    return np.dot(sense_emb, context_embed)/(np.linalg.norm(sense_emb)*np.linalg.norm(context_embed))

def extended_lesk(noun, sent):
    try: 
        # create context bag
        word_list = sent.copy()
        word_list.remove(noun)
        context_bag = set(word_list)
        context_bag = context_bag.difference(stopwords.words('english'))

        # get all word senses of sense bag
        syns = wn.synsets(noun)

        # get best sense
        best_sense = syns[0]
        max_score = 0

        for syn in syns:
            #get score of gloss of syns
            senses = []
            senses.append(syn)
            senses.extend(syn.hypernyms())
            senses.extend(syn.hyponyms())
            senses.extend(syn.member_holonyms())
            senses.extend(syn.root_hypernyms())

            flag = 0
            for sense in senses:
                gloss = set(word_tokenize(sense.definition())).difference(stopwords.words('english'))
                score = compute_score(gloss, context_bag)

                if score > max_score:
                    max_score = score
                    flag = 1

            if flag == 1:
                best_sense = syn

        return best_sense
    except:
        return None

def pagerank(word, sent, idx):
	# context 
	context = sent.copy()

	j=0
	end_prev = 0
	start_prev = 0
	word_start=0
	word_end = 0
	l = 2
	start_idx = max(idx-l,0)
	end_idx = min(idx+l+1,len(context))
	dict_nodes={}
	nodes=[]
	li=[]

		
	for i in range(start_idx,end_idx):
		ws = wn.synsets(context[i])
		start_prev_new = j
		for s in ws:
			nodes.append(j)
			dict_nodes[j]={'word':context[i],'sense':s}
			j+=1
		start_prev = start_prev_new
		end_prev = j
		li.append([start_prev,end_prev])
		if context[i] == word :
			word_start=start_prev
			word_end=end_prev
	edges = np.zeros([len(nodes),len(nodes)])
	sum_1 = [0 for i1 in range(len(nodes))]
	rank = [0 for i1 in range(len(nodes))]
	for w in range(len(li)-1):
		for m in range(li[w][0],li[w][1]):
			for n in range(li[w+1][0],li[w+1][1]):
				stopwords_set = set(stopwords.words('english'))
				sense1= set(word_tokenize(dict_nodes[m]['sense'].definition())).difference(stopwords_set)
				sense2= set(word_tokenize(dict_nodes[n]['sense'].definition())).difference(stopwords_set)
				edges[m][n]=compute_score(sense1,sense2)
				sum_1[m]+=edges[m][n]

	d=0.7
	while(True):
		new_rank = [0 for i1 in range(len(nodes))]
		for i1 in range(len(nodes)):
			for i2 in range(len(nodes)):
				if edges[i1][i2]!=0 and sum_1[i2]!=0:
					new_rank[i1]+=((rank[i2]/sum_1[i2])*edges[i1][i2])
			new_rank[i1]=new_rank[i1]*d+(1-d)/len(nodes)
		diff = 0
		for i1 in range(len(rank)):
			diff +=abs(rank[i1]-new_rank[i1])
		if diff < 1e-5:
			break
		rank = new_rank

		
	sense_rank_w=[]
	dict_sense={}
	for i in range(word_start,word_end):
		sense_rank_w.append(rank[i])
		dict_sense[i-word_start]=dict_nodes[i]['sense']
	return dict_sense[np.argmax(np.array(sense_rank_w))]

def main():
    print("\nChoose WSD Algorithm (extended lesk, pagerank), type exit to exit GUI")

    while(True):
        algo = input("\nAlgorithm: ")
        if algo == "exit":
            print("Exiting...")
            break
        
        if algo == "pagerank":
            sent = word_tokenize(input("Input sentence: "))
            word = input("Word you want to disambiguate: ")
            index = int(input("Index of chosen word in sentence: "))
            print("Predicted sense:",pagerank(word,sent,index).definition())

        elif algo == "extended lesk":
            sent = word_tokenize(input("Input sentence: "))
            word = input("Word you want to disambiguate: ")
            print("Predicted sense:",extended_lesk(word,sent).definition())

        else:
            print("Choose among pagerank and extended lesk")

if __name__ == "__main__":
    main()