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
from sklearn import linear_model

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


class WacGen():

    def __init__(self,google=True,saia=True,groundsplit=False):

        self.basedir = './SAIA_Data/benchmark/saiapr_tc-12'
        
        referit_path = './ReferitData/RealGames.txt'
        self.refdf = get_refexp(referit_path)

        if groundsplit:
            with gzip.open('./InData/Splits/ground-truth-split.pklz', 'r') as f:
                traintest = pickle.load(f)
        else:
            with gzip.open('./InData/Splits/90-10-split.pklz', 'r') as f:
                traintest = pickle.load(f)


        self.testfilelist = traintest[1][0]  # test is second in tuple
        self.testdf = make_subdf(self.refdf, self.testfilelist)

        self.trainfilelist = traintest[0][0]
        self.traindf = make_subdf(self.refdf, self.trainfilelist)

        #sentences = [sent.split() for sent in self.traindf.refexp]
        #self.vecmodel = word2vec.Word2Vec(sentences, size=12, sg=2, window=3, min_count=2, workers=2, iter=4)


        feat_path = './InData/Features/features.mat'
        featmat = scipy.io.loadmat(feat_path)
        self.X = featmat['X']


        if google and saia:
            base = './InData/Models/01-400gsaiafr7'
            Xg = np.load('./InData/googLefeats_pos_saia.npz')
            self.Xg = Xg['arr_0']

        elif google:
            if groundsplit:
                base = './InData/Models/groundsplit_400gfr7'
            else:
                base = './InData/Models/02-400gfr7'
            Xg = np.load('./InData/Features/googLefeats_pos.npz')
            self.Xg = Xg['arr_0']
            
        else:
            if groundsplit:
                base = './InData/Models/groundsplit_400saiafr7'
            else:
                base = './InData/Models/02-mn50'

        with open(os.path.join(base, 'params.txt'), 'r') as f:
            params = f.readlines()
        with gzip.open(os.path.join(base, 'wordclassifiers_0.mod'), 'r') as f:
            self.word_classifier = pickle.load(f)

                
        print "Loaded classifiers", params

        label_list = './InData/wlist.txt'
        self.label_index = {}
        self.index_label = {}

        with open(label_list, 'r') as f:
            for line in f.readlines():
                token = line.split()
                if len(token) > 1:
                    self.label_index[token[1]] = int(token[0])
                    self.index_label[int(token[0])] = token[1]

        #print self.index_label.keys()

        self.multi_label = []
        self.make_multi_label()

        # #print self.multi_label[0:10]

        self.label_vocab = {}
        self.label_filevocab = {}
        self.label_total = Counter()
        #self.make_label_vocab()

        with open('./InData/Splits/labelvocab.pklz', 'r') as f:
            (self.label_vocab,self.label_filevocab,self.label_total) = pickle.load(f)

        self.stop_words = ['and','of','to','at','on','that','very','next','has',\
        'it','thing','just','or','us','is','with','in','the','from','his','her','but']

        self.google = google

        self.make_lm()
        self.make_rev_lm()


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

    def make_rev_lm(self):

        bi_vocab = {}
        end_vocab = Counter()

        total = len(self.traindf.refexp)
        for s in list(self.traindf.refexp):
            utt = [w for w in s.split() if w in self.word_classifier]
            utt.reverse()
    
            for x in range(len(utt)):
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
            
            
        self.rev_bigrams = {}
        for w in bi_vocab:
            self.rev_bigrams[w] = Counter()
            for w2 in bi_vocab[w]:
                try:
                    self.rev_bigrams[w][w2] = bi_vocab[w][w2]/self.vocab[w]
                except:
                    print "word", w, "not in vocab"
        
        self.rev_end_probs = Counter()
        for w in end_vocab:
            self.rev_end_probs[w] = end_vocab[w]/self.vocab[w]


    def make_multi_label(self):

        featdf = pd.DataFrame(self.X[:,[0,1,-1]], columns=['file', 'region', 'label'])

        this_file = -1
        
        lr_dict = {}
        for _, row in featdf.iterrows():
            if row['file'] != this_file: # we have seen all we are going to see of previous file
                if len(lr_dict.keys()) != 0:

                    repeated = [(label, regions) for label, regions in lr_dict.items() if len(regions) >= 1]
                    if len(repeated) > 0:  # actually, we take all, not just repeated
                        self.multi_label.append((int(this_file), repeated))
                    repeated = []
                lr_dict = {}
                this_file = row['file']
            regions = lr_dict.get(int(row['label']), [])
            regions.append(int(row['region']))
            lr_dict[int(row['label'])] = regions



    def make_label_vocab(self):

    
        for fileid, repeats in self.multi_label:
            if fileid not in self.testfilelist:
                for label_index, regions in repeats:
        
                    #print repeats
                    this_label = self.index_label[label_index]
            
                    if this_label not in self.label_vocab:
                        self.label_vocab[this_label] = Counter()
                        self.label_filevocab[this_label] = defaultdict(list)
                    
                    for regionid in regions:
                        reflist = list(self.refdf[(self.refdf['file'] == fileid) & \
                            (self.refdf['region'] == regionid)]['refexp'])
                    
                        for l in reflist:
                            if not is_relational(l):
                                for word in l.split():
                                    if word in self.word_classifier:
                                        self.label_vocab[this_label][word] += 1
                                        self.label_filevocab[this_label][word].append((fileid,regionid))

        for l in self.label_vocab:
            self.label_total[l] = sum([p for (w,p) in self.label_vocab[l].items()])





    def get_features(self,fileid,regionid):

        res = []

        if self.google:
            f = self.Xg[np.logical_and(self.Xg[:,0] == fileid, self.Xg[:,1] == regionid)][:,2:-1]
        else:
            f = self.X[np.logical_and(self.X[:,0] == fileid, self.X[:,1] == regionid)][:,2:-1]

        if len(f) > 0:
            res = f[0].reshape(1,-1)
    #if len(f) > 1:
        #print "multiple feature vectors, are they identical?"
        #print f
    #    f = f[0]
        return res

    def get_gold_utt(self,fileid,regionid):

        
        return list(self.refdf[(self.refdf['file'] == fileid) & (self.refdf['region'] == regionid)]['refexp'])

    def get_dist_eval_data(self,filterstop=False):

        eval_list = []

        for fileid, repeats in self.multi_label:
            if fileid in self.testfilelist:
                for label_index,regions in repeats:
                    if len(regions) > 1:
                        for regionid in regions:
                
                            gold_utt = self.get_gold_utt(fileid,regionid)

                            if len(gold_utt) > 0: 

                                if not is_relational(gold_utt[0]):
                    
                                    gold_utt = gold_utt[0].split(' ')
                                    gold_utt = [w for w in gold_utt if w in self.word_classifier]
                                    if filterstop:
                                        gold_utt = [w for w in gold_utt if not w in self.stop_words]
                                    eval_list.append((fileid,regionid,regions,self.index_label[label_index],gold_utt))
                            else:
                    #print "no gold utt found", fileid,regionid
                                break

        return eval_list

    def get_eval_data(self,filterstop=False):

        eval_list = []

        for fileid, repeats in self.multi_label:
            if fileid in self.testfilelist:
                for label_index,regions in repeats:
                    
                    for regionid in regions:
                
                        gold_utt = self.get_gold_utt(fileid,regionid)

                        if len(gold_utt) > 0: 

                            if not is_relational(gold_utt[0]):
                    
                                gold_utt = gold_utt[0].split(' ')
                                gold_utt = [w for w in gold_utt if w in self.word_classifier]
                                if filterstop:
                                    gold_utt = [w for w in gold_utt if not w in self.stop_words]

                                if len(gold_utt) > 0:
                                    eval_list.append((fileid,regionid,regions,self.index_label[label_index],gold_utt))
                        else:
                    #print "no gold utt found", fileid,regionid
                            break

        return eval_list
        
    def get_eval_measures(self,eval_list,gen_list):

        prec = 0
        recall = 0
        total_overlap = 0
        total_words = 0

        dist_overlap = 0
        dist_words = 0

        nodist_overlap = 0
        nodist_words = 0

        total = 0

        for i in range(0,len(eval_list)):
            gold_utt = set(eval_list[i][4])
            gen_utt = set(gen_list[i])

            overlap = gen_utt & gold_utt
                
            prec += len(overlap)/len(gen_utt)
            recall += len(overlap)/len(gold_utt)
            total_overlap += len(overlap)
            total_words += len(gold_utt)

            if len(eval_list[i][2]) > 1:
                dist_overlap += len(overlap)
                dist_words += len(gen_utt)
            else:
                nodist_overlap += len(overlap)
                nodist_words += len(gen_utt)


            total += 1
          
        print "Accuracy for distractor regions", dist_overlap/dist_words 
        print "Accuracy for others", nodist_overlap/nodist_words 
                    
        return prec,recall,total,total_overlap/total_words
                
    def get_av_distance(self,some_words,this_word):

        total = 0

        for other_word in some_words:

            other_vec = self.word_classifier[other_word].raw_coef_[0]
            this_vec = self.word_classifier[this_word].raw_coef_[0]
            total += scipy.spatial.distance.canberra(this_vec,other_vec)
                    
        return total/len(some_words)

    def get_len_classifiers(self):
    
    
        X_train = {1:[],2:[],3:[],4:[],5:[]}
        Y_train = {1:[],2:[],3:[],4:[],5:[]}
        
                
        for _, row in self.traindf.iterrows():
            if is_relational(row['refexp']):
                continue
            expr = set([word for word in row['refexp'].split()])
            
            res = self.get_features(row['file'],row['region'])
            if len(res) > 0:
                regionlist = self.traindf[self.traindf['file'] == row['file']]['region']
                res = np.append(res,len(regionlist))

                
            #X_train.append(gen.get_features(row['file'],row['region']))
                if len(expr) == 1:
                    X_train[1].append(res)
                    Y_train[1].append(1)
                elif Y_train[1].count(0) < 50000:
                    X_train[1].append(res)
                    Y_train[1].append(0)
                
                    if len(expr) == 2:
                        X_train[2].append(res)
                        Y_train[2].append(1)
                    elif Y_train[2].count(0) < 30000:
                        X_train[2].append(res)
                        Y_train[2].append(0)
                    
                        if len(expr) == 3:
                            X_train[3].append(res)
                            Y_train[3].append(1)
                        elif Y_train[3].count(0) < 30000:
                            X_train[3].append(res)
                            Y_train[3].append(0)
                        
                            if len(expr) == 4:
                                X_train[4].append(res)
                                Y_train[4].append(1)
                            elif Y_train[4].count(0) < 30000:
                                X_train[4].append(res)
                                Y_train[4].append(0)
                            
                                if len(expr) == 5:
                                    X_train[5].append(res)
                                    Y_train[5].append(1)
                                elif Y_train[5].count(0) < 30000:
                                    X_train[5].append(res)
                                    Y_train[5].append(0)
                        
                        
        len_classifier = {}
        for key in X_train:
            print "Train len", key
            endregr = linear_model.LogisticRegression(penalty='l1')
            endregr.fit(X_train[key], Y_train[key])
            len_classifier[key] = endregr
        
        return X_train,Y_train,len_classifier

    def generate_beam_plen(self,len_class,infile,regiontarget):
            
        target_test = self.get_features(infile,regiontarget)
        
        lentarget = 3
        
        feat = list(target_test[0])
        if len(feat) > 0:
            regionlist = self.testdf[self.testdf['file'] == infile]['region']
            feat.append(len(regionlist))
            feat = np.array(feat).reshape(1,-1)
        
            for l in range(1,6):
                #print l#, len_class[l]
                res = len_class[l].predict(feat)
                #print res
                if res:
                    lentarget = l
                    break
          
        #print lentarget
        
        
        word_fits = Counter()
        for word in self.word_classifier:                
            word_fits[word] = log(self.word_classifier[word].predict_proba(target_test)[:, 1][0])
                
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
                    
                        if not next_word in utt:
                            next_score += word_fits[next_word] 
                            next_score += log(self.bigrams[prev_word][next_word])
                        #if uttlen == lentarget-1:
                        #    next_score += gen.end_probs[next_word]
                            
                            next_beam.append((next_score,utt+[next_word]))
            
            
            beam = sorted(next_beam,reverse=True)[0:50]
            #print beam
            uttlen += 1

            
        #regionlist = gen.testdf[gen.testdf['file'] == infile]['region']
        #print "Regions",regionlist
        #print gen.testdf[gen.testdf['file'] == fileid]
        
        for (score,utt) in beam:
            if self.end_probs[utt[-1]] > 0.15:
                return utt
            
        return beam[0][1]

    def generate_beam(self,infile,regiontarget,lentarget):
            
        target_test = self.get_features(infile,regiontarget)
        word_fits = Counter()
        
        for word in self.word_classifier:      
            #print target_test.shape          
            word_fits[word] = log(self.word_classifier[word].predict_proba(target_test)[:, 1][0])
                
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
                    
                        if not next_word in utt:
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

  