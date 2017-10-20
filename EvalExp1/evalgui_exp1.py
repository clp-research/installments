# -*- coding: utf-8 -*-
from __future__ import division

import Tkinter as tk
from PIL import Image, ImageTk

import pandas as pd
# from collections import Counter
import scipy.io
import numpy as np
# from sklearn import linear_model
# from random import shuffle

import cPickle as pickle

import re

from scipy import misc
from operator import itemgetter

import random
import time
 
from optparse import OptionParser



class EvalApp(object):


    def __init__(self, parent,tnumber=16):
        """Constructor"""

        tnumber = tnumber - (tnumber%4)
        print "Number of games", tnumber

        testpath = '../SAIA_Data/test_set_ground_truth'
        atest = pd.read_csv('./exp1_ground_truth_generated.csv')

        self.logname = 'final_logs/log_'+str(tnumber)+'_'+str(time.time())+'.txt'
        self.logfile = open(self.logname,'w')
        print 'log_'+str(time.time())+'.txt'
        print >>self.logfile,str(time.clock()),'./exp1_ground_truth_generated.csv'

        afiles = [(row['file'],row['region'],row['Testset']) for (_,row) in atest[atest['Testset'] == 'A'].iterrows()]
        bfiles = [(row['file'],row['region'],row['Testset']) for (_,row) in atest[atest['Testset'] == 'B'].iterrows()]
        cfiles = [(row['file'],row['region'],row['Testset']) for (_,row) in atest[atest['Testset'] == 'C'].iterrows()]

        random.shuffle(afiles)
        random.shuffle(bfiles)
        random.shuffle(cfiles)

        i = int(tnumber / 3)
        self.testfiles = afiles[:i] + bfiles[:i] + cfiles[:(i+tnumber%3)]
        print "Testfiles",len(self.testfiles)
        #print self.testfiles

        i = int(tnumber/4)
        testsystem = ['gold']*i + ['saia']*i + ['google']*i + ['google_glen']*i
        random.shuffle(testsystem)
        

        #for x in enumerate(self.testfiles):
        #    print x

        self.testexpr = [list(atest[(atest['file'] == fileid) & (atest['region'] == regionid)][testsystem[r]])[0] \
                        for (r,(fileid,regionid,_)) in enumerate(self.testfiles) ]


        self.tindex = 0
        self.root = parent
        self.basedir = '../SAIA_Data/benchmark/saiapr_tc-12'
        self.trials_left = 3


        t = "Play the ReferIt Game!"
        tk.Label(self.root, text=t,font=('Helvetica',14)).pack()


        t2 = "Please select:"
        tk.Label(self.root, text=t2,font=('Helvetica',14)).pack()

        this_file = 8756
        this_region = 10


        self.testfiles = [(this_file,this_region,'Test')] + self.testfiles
        self.testsystem = ['test'] + testsystem
        print len(testsystem),testsystem
        self.testexpr = ['guy in the middle in front'] + self.testexpr
        #print self.testexpr

        self.clicks = [0]*len(self.testexpr)
        self.lastclick = [0]*len(self.testexpr)
        self.nextclick = [0]*len(self.testexpr)
        self.success = [0]*len(self.testexpr)

        print >>self.logfile,str(time.clock()),"open", this_file,this_region


        t3 = 'guy in the middle in front'
        self.refVariable = tk.StringVar()
        tk.Label(self.root, textvariable=self.refVariable,font=('Helvetica',20),bg="gray").pack()
        self.refVariable.set(t3)

        print >>self.logfile,str(time.clock()),"RE", t3


        if len(str(this_file)) == 5:
            directory = str(this_file)[:2]
        elif len(str(this_file)) < 4:
            directory = '00'
        else:
            directory = '0' + str(this_file)[0]
        self.imgpath = self.basedir + '/' + directory + '/images/' + str(this_file) + '.jpg'
#img = misc.imread(path)

        maskpath = self.basedir + '/' + directory + '/segmentation_masks/' + str(this_file) + '_' + str(this_region) + '.mat'
        this_mask = scipy.io.loadmat(maskpath)
        self.this_mask = this_mask['segimg_t']

        xlen = self.this_mask.shape[1]
        ylen = self.this_mask.shape[0]

        self.tkimage = ImageTk.PhotoImage(Image.open(self.imgpath).resize((xlen,ylen)))
        #self.tkimage = ImageTk.PhotoImage(Image.open(self.imgpath))
        self.ilabel = tk.Label(self.root, image=self.tkimage)
        self.ilabel.bind("<Button-1>", self.callback)
        #self.ilabel.grid(column=0, row=0, columnspan=2, sticky='EW')
        self.ilabel.pack()

        self.feedbackVariable = tk.StringVar()
        tk.Label(self.root, textvariable=self.feedbackVariable,font=('Helvetica',20),fg="white",bg="blue").pack()
        self.feedbackVariable.set("...")

        self.trialVariable = tk.StringVar()
        tk.Label(self.root, textvariable=self.trialVariable,font=('Helvetica',14),fg="white",bg="blue").pack()
        self.trialVariable.set("Trials left:" + str(self.trials_left))

        print >>self.logfile,str(time.clock()),"Trials left:" + str(self.trials_left)


        
        stopbutton = tk.Button(self.root,text="Stop",command=self.hide)
        stopbutton.pack(side=tk.LEFT)

        nextbutton = tk.Button(self.root,text="Next",command=self.OnButtonClick)
        nextbutton.pack(side=tk.RIGHT)

    #----------------------------------------------------------------------
    def OnButtonClick(self):

        self.tindex += 1
        self.nextclick[self.tindex] = time.clock()

        self.trials_left = 3
        self.trialVariable.set("Trials left:" + str(self.trials_left))

        fileid = self.testfiles[self.tindex][0]
        this_region = self.testfiles[self.tindex][1]

        print >>self.logfile,str(time.clock()),"open", fileid,this_region


        if len(str(fileid)) == 5:
            directory = str(fileid)[:2]
        elif len(str(fileid)) < 4:
            directory = '00'
        else:
            directory = '0' + str(fileid)[0]
        self.imgpath = self.basedir + '/' + directory + '/images/' + str(fileid) + '.jpg'
#img = misc.imread(path)

        maskpath = self.basedir + '/' + directory + '/segmentation_masks/' + str(fileid) + '_' + str(this_region) + '.mat'
        this_mask = scipy.io.loadmat(maskpath)
        self.this_mask = this_mask['segimg_t']

        xlen = self.this_mask.shape[1]
        ylen = self.this_mask.shape[0]

        self.tkimage = ImageTk.PhotoImage(Image.open(self.imgpath).resize((xlen,ylen)))
        self.ilabel.config(image=self.tkimage)
        

        t3 = self.testexpr[self.tindex]
        self.refVariable.set(t3)
        self.feedbackVariable.set("...")

        print >>self.logfile,str(time.clock()),"RE", t3

        print self.tindex, len(self.testfiles)
        
        



    #----------------------------------------------------------------------
    def callback(self,event):
        self.ilabel.focus_set()
        print "clicked at", event.x, event.y, self.this_mask[event.y][event.x] != -1

        print >>self.logfile,str(time.clock()),"clicked at", event.x, event.y, self.this_mask[event.y][event.x] != -1

        pos = ['Hurray!','Superb!','Great!','Well done!','Fantastic!','Very good.','Correct.']
        neg = ["Not quite. Try again?", "Sorry, please try again!","Hm, not really.","Sorry, I want a different one."]

        if self.this_mask[event.y][event.x] != -1:
            self.feedbackVariable.set(random.choice(pos))
            self.trialVariable.set("Please press next to continue.")
            print >>self.logfile,str(time.clock()),"Correct. Should press next."

            self.clicks[self.tindex] += 1
            self.success[self.tindex] = True
            self.lastclick[self.tindex] = time.clock()

            if self.tindex == len(self.testfiles)-1:
                print "Hello?"
                self.hide()

        else:
            self.trials_left -= 1
            self.trialVariable.set("Trials left:" + str(self.trials_left))

            if self.trials_left > 0:
                self.feedbackVariable.set(random.choice(neg))
                print >>self.logfile,str(time.clock()),"Incorrect. Trials left:" + str(self.trials_left)

            elif self.tindex == len(self.testfiles)-1:
                print "Hello?"
                self.hide()

            else:
                self.feedbackVariable.set("Nevermind. Please press next.")
                print >>self.logfile,str(time.clock()),"Incorrect. No Trials left."

            self.clicks[self.tindex] += 1
            self.success[self.tindex] = False
            self.lastclick[self.tindex] = time.clock()




    #----------------------------------------------------------------------
    def hide(self):
        """"""
        tfiles = [x for (x,y,z) in self.testfiles]
        tregions = [y for (x,y,z) in self.testfiles]
        tsets = [z for (x,y,z) in self.testfiles]


        self.feedbackVariable.set("THANK YOU!")
        self.trialVariable.set("This was a good piece of work.")

        print len(tfiles),len(tregions),len(tsets),len(self.clicks),len(self.success),len(self.lastclick)
        logframe = pd.DataFrame({'file':tfiles,'region':tregions,'testset':tsets,'refexp':self.testexpr,'system':self.testsystem,\
            'clicks':self.clicks,'success':self.success,'lastClick':self.lastclick,'start':self.nextclick})
        logframe.to_csv(self.logname[:-4]+"_df.csv")

        time.sleep(10)

        self.root.destroy()



#----------------------------------------------------------------------
if __name__ == "__main__":

    oparser = OptionParser()
    oparser.add_option("-n","--number",dest="n",help="Number of games")
    (options,args) = oparser.parse_args()


    root = tk.Tk()
   # default_font = tkFont.nametofont("TkDefaultFont")
   # default_font.configure(size=12)

   # root.geometry("1300x600+0+0")
    app = EvalApp(root,tnumber=int(options.n))
    root.mainloop()
