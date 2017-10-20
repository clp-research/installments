from __future__ import division

from collections import Counter,defaultdict
from math import log

import numpy as np
import scipy.io
import cPickle as pickle
import gzip
import re
import os.path
import pandas as pd
import gensim
from gensim.models import word2vec
from sklearn import linear_model
import argparse

import sys
sys.path.append('../../v1')
sys.path.append('../../v2/Code')

rel_list = ['below',
            'above',
            'between',
            'not',
            'behind',
            'under',
            'underneath',
            'front of',
            'right of',
            'left of',
            'ontop of',
            'next to',
            'middle of']

def get_refexp(path):
    refdf = pd.read_csv(path, sep='~',
                        names=['ID', 'refexp', 'regionA', 'regionB'])

    refdf['file'] = refdf['ID'].apply(lambda x:
                                      int(x.split('.')[0].split('_')[0]))
    refdf['region'] = refdf['ID'].apply(lambda x:
                                        int(x.split('.')[0].split('_')[1]))

    # - make preprocessing function an option as well?
    refdf['refexp'] = preproc_vec(refdf['refexp'])

    return refdf

def preproc(utterance):
    utterance = re.sub('[\.\,\?;]+', '', utterance)
    return utterance.lower()
preproc_vec = np.vectorize(preproc)

def make_subdf(df, filelist):
    files_df = pd.DataFrame({'file': filelist})
    return pd.merge(df, files_df)

def is_relational(refexp):
    # rel_list is a global variable
    for rel in rel_list:
        if rel in refexp:
            return True
    return False



class WacGenIA():

    def __init__(self):

        self.basedir = './SAIA_Data/benchmark/saiapr_tc-12'
        referit_path = './ReferitData/RealGames.txt'
        self.refdf = get_refexp(referit_path)

        with gzip.open('./InData/Splits/ground-truth-split.pklz', 'r') as f:
            traintest = pickle.load(f)

        goog_path = './InData/Features/googLefeats_betterpos.npz'
        X = np.load(goog_path)
        self.X = X['arr_0']

        base = './InData/Models/betterpos-groundsplit-400gfr7'
        with open(os.path.join(base, 'params.txt'), 'r') as f:
            params = f.readlines()
        with gzip.open(os.path.join(base, 'wordclassifiers_0.mod'), 'r') as f:
            self.word_classifier = pickle.load(f)

        print "Loaded word classifiers"

        self.testpath = './SAIA_Data/test_set_ground_truth'
        
        atest = pd.read_csv(self.testpath+'/ground_truth_a.csv')
        btest = pd.read_csv(self.testpath+'/ground_truth_b.csv')
        ctest = pd.read_csv(self.testpath+'/ground_truth_c.csv')
        atest['Testset'] = 'A'
        btest['Testset'] = 'B'
        ctest['Testset'] = 'C'

        alltest = atest.append(btest)
        alltest = alltest.append(ctest)
        alltest['prev_file'] = alltest['Image Number'].shift()
        alltest['same'] = alltest['Image Number'] == alltest['prev_file']

        self.testdf = alltest[alltest['same'] == False]
        files = [int(i.split('_')[0]) for i in self.testdf['Image Number']]
        regions = [int(i.split('_')[1][:-4]) for i in self.testdf['Image Number']]
        self.testdf['file'] = files
        self.testdf['region'] = regions

        print "Loaded testfiles", len(self.testdf)

        self.trainfilelist = traintest[0][0]
        self.traindf = make_subdf(self.refdf, self.trainfilelist)

        self.make_lm()


    def make_lm(self):

        self.vocab = Counter()
        bi_vocab = {}
        end_vocab = Counter()

        total = len(self.traindf.refexp)
        for s in list(self.traindf.refexp):
            utt = [w for w in s.split() if w in self.word_classifier]
    
            for x in range(len(utt)):
                self.vocab[utt[x]] += 1
                if x == 0:
                    if not 'start' in bi_vocab:
                        bi_vocab['start'] = Counter()
                    bi_vocab['start'][utt[x]] += 1
                else:
                    if not utt[x-1] in bi_vocab:
                        bi_vocab[utt[x-1]] = Counter()
                    bi_vocab[utt[x-1]][utt[x]] += 1
            
                if x == len(utt)-1:
                    end_vocab[utt[x]] += 1
            
    
        self.vocab['start'] = total
        
        self.bigrams = {}
        for w in bi_vocab:
            self.bigrams[w] = Counter()
            for w2 in bi_vocab[w]:
                self.bigrams[w][w2] = bi_vocab[w][w2]/self.vocab[w]
        
        self.end_probs = Counter()
        for w in end_vocab:
            self.end_probs[w] = end_vocab[w]/self.vocab[w]


    def get_features(self,fileid,regionid):

        res = []
        f = self.X[np.logical_and(self.X[:,0] == fileid, self.X[:,1] == regionid)][:,2:-1]

        if len(f) > 0:
            res = f[0]
    #if len(f) > 1:
        #print "multiple feature vectors, are they identical?"
        #print f
    #    f = f[0]
        return res


    def generate_beam(self,wclassifier,infile,regiontarget,lentarget,force_loc=0):
            
        target_test = [self.get_features(infile,regiontarget)]
        word_fits = Counter()
        
        for word in wclassifier:
            #if word == "sky":
            #    print "sky"             
            word_fits[word] = log(wclassifier[word].predict_proba(target_test)[:, 1][0])
                
        beam = [(0,['start'])]
        uttlen = 0
        
        while uttlen < lentarget:
            
            #print uttlen,lentarget
            
            next_beam = []
            for (score,utt) in beam:
                
                prev_word = utt[-1]
                
                if prev_word in self.bigrams:
                
                    for next_word in self.bigrams[prev_word]:
                        next_score = score
                    
                        if (not next_word in utt) and (next_word in wclassifier): # added second if-condition
                            next_score += word_fits[next_word] 
                            next_score += log(self.bigrams[prev_word][next_word])
                            if uttlen == lentarget-1:
                                next_score += self.end_probs[next_word]
                            
                            next_beam.append((next_score,utt+[next_word]))
            
            
            beam = sorted(next_beam,reverse=True)[0:50]
            #print beam
            uttlen += 1


        for (score,utt) in beam:
            if self.end_probs[utt[-1]] > 0.15:
                return utt

            
        return beam[0][1]


    def make_ia_classifiers(self):

    	label_list = './InData/wlist.txt'
        self.label_index = {}
        self.index_label = {}

        with open(label_list, 'r') as f:
            for line in f.readlines():
                token = line.split()
                if len(token) > 1:
                    self.label_index[token[1]] = int(token[0])
                    self.index_label[int(token[0])] = token[1]

    	categories = set(self.label_index.keys()) | set(['kid','girl','boy','people','men','women',\
                                                      'pot','sign','face','head','spiders','bushes',\
                                                      'bldg','leaves'])

    	atest = pd.read_csv(self.testpath+'/ground_truth_a.csv')
        categories = categories | set(list(atest['Entry-Level']))
        #categories = categories | set(list(btest['Entry-Level']))... these contain noisy categories
        #categories = categories | set(list(ctest['Entry-Level']))
        categories = categories | set([str(c)+'s' for c in categories])
        categories = categories - set(['womans','one'])

        sizes = ['big','huge','large','little','long','short','small','tall','tiny']
        colors = ['blue','red','green','yellow','white','black','gray','grey','pink',\
        'purple','rose','orange','brown','tan','dark']
        locations = ['left','right','bottom','top','middle','side','corner','front','background',\
        'very','far','center','in','on','the','thing','anywhere','upper','lower','leftmost','rightmost']



        excluded = []
        self.word_classifier_a1 = {}
        self.word_classifier_a2 = {}
        self.word_classifier_a3 = {}
        self.word_classifier_type = {}

        for w in self.word_classifier:
            if w in categories:
                self.word_classifier_a3[w] = self.word_classifier[w]
                self.word_classifier_a2[w] = self.word_classifier[w]
                self.word_classifier_type[w] = self.word_classifier[w]
            elif w in sizes:
                self.word_classifier_a3[w] = self.word_classifier[w]
                #self.word_classifier_a2[w] = self.word_classifier[w]
            elif w in colors:
                self.word_classifier_a3[w] = self.word_classifier[w]
                #self.word_classifier_a2[w] = self.word_classifier[w]
            elif w in locations:
                self.word_classifier_a3[w] = self.word_classifier[w]
                self.word_classifier_a2[w] = self.word_classifier[w]
                self.word_classifier_a1[w] = self.word_classifier[w]
            else:
                excluded.append(w)

        print "Excluded classifiers"," ,".join(excluded)
        print "A1 classifiers", len(self.word_classifier_a1)
        print "A2 classifiers", len(self.word_classifier_a2)
        print "A3 classifiers", len(self.word_classifier_a3)
        print "Type classifiers"," ,".join(self.word_classifier_type.keys())

        #print "Number of attribute classifiers", len(self.word_classifier_att)


    def generate_ia_distractors(self):

        self.file_region_att = {}
        for (_,row) in self.testdf.iterrows():

            tfile = row['file']
            if not tfile in self.file_region_att:
                tregions = self.refdf[self.refdf['file'] == tfile]['region']
                self.file_region_att[tfile] = {}

                for tregion in set(list(tregions)):
                    try:
                        gen2 = self.generate_beam(self.word_classifier_a1,tfile,tregion,2)
                        gen3 = self.generate_beam(self.word_classifier_a2,tfile,tregion,4)
                        gen4 = self.generate_beam(self.word_classifier_a3,tfile,tregion,6)
                        self.file_region_att[tfile][tregion] = [gen2,gen3,gen4]
                    except:
                        print "Could not generate",tfile,tregion


    def generate_type_distractors(self):

        self.file_region_type = {}
        for (_,row) in self.testdf.iterrows():

            tfile = row['file']
            #print tfile
            if not tfile in self.file_region_type:
                tregions = self.refdf[self.refdf['file'] == tfile]['region']
                self.file_region_type[tfile] = {}

                for tregion in set(list(tregions)):
                    target_test = [self.get_features(tfile,tregion)]

                    if len(target_test[0]) > 0:
                    	word_fits = Counter()
                    	for word in self.word_classifier_type:          
                        	word_fits[word] = log(self.word_classifier_type[word].predict_proba(target_test)[:, 1][0])

                    	self.file_region_type[tfile][tregion] = [genw for (genw,_) in word_fits.most_common(3)]

                    else:
                    	print "No Features for", tfile,tregion


    def generate_label_hedges(self,uttlist,hedgelist):

        hedged_utts = []
        for u in uttlist:
            hedged_u = []
            for w in u:
                if w in self.word_classifier_type:
                    hedged_u.append(' or '.join(hedgelist))
                    break
                else:
                    hedged_u.append(w)

            hedged_u += u[len(hedged_u):]
            hedged_utts.append(hedged_u)     
         
        return hedged_utts

    def generate_ia(self):

    	utt_list = []


        for (_,row) in self.testdf.iterrows():
    
            tfile = row['file']
            tregion = row['region']
    
            if tfile in self.file_region_att:
                if tregion in self.file_region_att[tfile]:
            
                    tutts = self.file_region_att[tfile][tregion]
                    tnoun = self.file_region_type[tfile][tregion][0]
                    tnoun2 = self.file_region_type[tfile][tregion][1]

                    hutt1,hutt2 = self.generate_label_hedges((tutts[1],tutts[2]),(tnoun,tnoun2))

                    dregions = [r for r in self.file_region_att[tfile] if not r == tregion]

                    dist2 = [r for r in dregions if self.file_region_att[tfile][r][0] == tutts[0]]
                    dist3 = [r for r in dregions if self.file_region_att[tfile][r][1] == tutts[1]]
                    dist4 = [r for r in dregions if self.file_region_att[tfile][r][2] == tutts[2]]

                    if len(dist2) == 0:
                    	utt_list.append((tutts[0],tutts[1],tutts[2],hutt1,hutt2,tnoun+':'+tnoun2,'a1'))
                    elif len(dist3) == 0:
                    	utt_list.append((tutts[0],tutts[1],tutts[2],hutt1,hutt2,tnoun+':'+tnoun2,'a2'))
                    elif len(dist4) == 0:
                    	utt_list.append((tutts[0],tutts[1],tutts[2],hutt1,hutt2,tnoun+':'+tnoun2,'a3'))
                    else:
                    	utt_list.append((tutts[0],tutts[1],tutts[2],hutt1,hutt2,tnoun+':'+tnoun2,'a0'))
                    	print "No distinguishing att expr found!",tfile,tregion
                else:
                	utt_list.append(("","","","","","",""))
            else:
            	utt_list.append(("","","","","","",""))

        return utt_list


    def generate_type_context(self):

    	utt_list = []


        for (_,row) in self.testdf.iterrows():
    
            tfile = row['file']
            tregion = row['region']
    
            if tfile in self.file_region_loc:
                if tregion in self.file_region_loc[tfile]:
            
                    tutts = self.file_region_loc[tfile][tregion]
                    tnoun = self.file_region_type[tfile][tregion][0]
                    tnoun2 = self.file_region_type[tfile][tregion][1]
                    dregions = [r for r in self.file_region_loc[tfile] if not r == tregion]

                    distloc = [r for r in dregions if self.file_region_loc[tfile][r][0] == tutts[0]]
                    disttype = [r for r in dregions if self.file_region_type[tfile][r][0] == tnoun]

                    if len(distloc) == 0:
                    	utt_list.append((tutts[0],tutts[1]+['looks like',tnoun],tutts[2]+['could be a',tnoun2]))
                    elif len(disttype) == 0:
                    	utt_list.append((tutts[0]+['looks like',tnoun],tutts[1]+['could be a',tnoun2],tutts[2]))
                    else:
                    	utt_list.append((tutts[1]+['looks like',tnoun],tutts[2]+['could be a',tnoun2],tutts[0]))
                    	print "No distinguishing type expr found!",tfile,tregion
                else:
                	utt_list.append(("","",""))
            else:
            	utt_list.append(("","",""))

        return utt_list


    def has_attributes(self,utt):

    	loc = set(['left','right','bottom','top','middle','side','corner',\
                'front','background','center','anywhere'])
        color = set(['blue','red','green','yellow','white','black','gray','grey','pink','purple','rose','orange','brown','tan'])
        size = set(['large','small','big','huge','tiny'])

        if len(set(utt) & loc) > 0:
        	return True

        if len(set(utt) & color) > 0:
        	return True

        if len(set(utt) & size) > 0:
        	return True


        return False

    def has_loc_attributes(self,utt):

    	loc = set(['left','right','bottom','top','middle','side','corner',\
                'front','background','center','anywhere','far'])
        
        return len(set(utt) & loc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Learn the WAC models')
    parser.add_argument('--max', dest='max', action='store',type=int,default=None)
    args = parser.parse_args(sys.argv[1:])

    genia = WacGenIA()
    genia.make_ia_classifiers()


    if args.max:
        print "reducing n files to",args.max
        genia.testdf=genia.testdf.head(n=args.max)
        print len(genia.testdf)

    genia.generate_ia_distractors()
	#genia.generate_loc_distractors()
    genia.generate_type_distractors()

    att_l = genia.generate_ia()
	#type_l = genia.generate_type_context()

    genia.testdf['ref_a1'] = [" ".join(u[0][1:]) for u in att_l]
    genia.testdf['ref_a2'] = [" ".join(u[1][1:]) for u in att_l]
    genia.testdf['ref_a3'] = [" ".join(u[2][1:]) for u in att_l]
    genia.testdf['ref_h_a2'] = [" ".join(u[3][1:]) for u in att_l]
    genia.testdf['ref_h_a3'] = [" ".join(u[4][1:]) for u in att_l]
    genia.testdf['ref_h_nouns'] = [u[5] for u in att_l]
    genia.testdf['ref_gentype'] = [u[6] for u in att_l]

    gfile = '../OutData/Eval_Context/ground_truth_generated.csv'

    if args.max:
        gfile = '../OutData/Eval_Context/ground_truth_generated_'+str(args.max)+'.csv'

    genia.testdf.to_csv(gfile)


	#genia.testdf['ref_type1'] = [u[0] for u in type_l]
	#genia.testdf['ref_type2'] = [u[1] for u in type_l]
	#genia.testdf['ref_type3'] = [u[2] for u in type_l]

	#genia.testdf.to_csv('../OutData/Eval_Context/ground_truth_generated.csv')
 





