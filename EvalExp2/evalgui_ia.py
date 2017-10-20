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

        self.root = parent


        self.basedir = "../SAIA_Data/benchmark/saiapr_tc-12"
        #'/Users/sina/generation/SAIA_Data/benchmark/saiapr_tc-12'
        self.gendf = pd.read_csv('exp2_ground_truth_generated.csv')

        self.logname = 'final_logs/log_ia_'+str(tnumber)+'_'+str(time.time())+'.txt'
        self.logfile = open(self.logname,'w')
        print 'final_logs/log_ia_'+str(tnumber)+'_'+str(time.time())+'.txt'
        print >>self.logfile,str(time.clock()),'exp2_ground_truth_generated.csv'       

        self.testexpr = []
        self.testfiles = []
        tnumber = tnumber - (tnumber%4)
        print "Number of games", tnumber
        self.MakeTestExpressions(tnumber)


        self.tindex = 0
        self.trials_left = 3
        self.clicks = [0]*len(self.testexpr)
        self.lastclick = [0]*len(self.testexpr)
        self.nextclick = [0]*len(self.testexpr)
        self.success = [0]*len(self.testexpr)

        
        t = "Play the ReferIt Game!"
        tk.Label(self.root, text=t,font=('Helvetica',14)).pack()


        t2 = "Please select:"
        tk.Label(self.root, text=t2,font=('Helvetica',14)).pack()


        #t3 = 'guy in the middle in front'
        this_expr = self.testexpr[self.tindex][3-self.trials_left]
        self.refVariable = tk.StringVar()
        tk.Label(self.root, textvariable=self.refVariable,font=('Helvetica',20),bg="gray").pack()
        self.refVariable.set(this_expr)

        print >>self.logfile,str(time.clock()),"RE", this_expr

        #this_file = 8756
        #this_region = 10
        this_file = self.testfiles[self.tindex][0]
        this_region = self.testfiles[self.tindex][1]

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

        print >>self.logfile,str(time.clock()),"open", this_file,this_region

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
    def MakeTestExpressions(self,tnumber):


        afiles = [row for (_,row) in self.gendf[self.gendf['Testset'] == 'A'].iterrows()]
        bfiles = [row for (_,row) in self.gendf[self.gendf['Testset'] == 'B'].iterrows()]
        cfiles = [row for (_,row) in self.gendf[self.gendf['Testset'] == 'C'].iterrows()]

        random.shuffle(afiles)
        random.shuffle(bfiles)
        random.shuffle(cfiles)

        i = int(tnumber / 3)
        self.testrows = afiles[:i] + bfiles[:i] + cfiles[:(i+tnumber%3)]
        print "Testfiles",len(self.testfiles)

        self.testfiles = [(row['file'],row['region'],row['Testset']) for row in self.testrows]
        self.testref = []
        
        for row in self.testrows:
            self.testref.append(row['ref_gentype'])
            if row['ref_gentype'] == 'a1':
                self.testexpr.append((row['ref_a1'],row['ref_a2'],row['ref_h_a2']))
            elif row['ref_gentype'] == 'a2':
                self.testexpr.append((row['ref_a2'],row['ref_h_a2'],row['ref_h_a3']))
            else:
                self.testexpr.append((row['ref_a3'],row['ref_h_a3'],row['ref_a1']))

        self.testexpr = [('guy in the middle in front','guy in the middle','in front middle')] + self.testexpr
        self.testfiles = [(8756,10,'T')] + self.testfiles
        self.testref = ['test'] + self.testref



    #----------------------------------------------------------------------
    def OnButtonClick(self):

        self.tindex += 1
        

        print "Tindex", self.tindex

        if self.tindex == len(self.testfiles):
            print "HIDE"
            self.feedbackVariable.set("THANK YOU!")
            self.trialVariable.set("This was a good piece of work.")

            
            self.hide()

        else:

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
            

            t3 = self.testexpr[self.tindex][3-self.trials_left]
            self.refVariable.set(t3)

            self.feedbackVariable.set("...")

            print >>self.logfile,str(time.clock()),"RE", t3,self.testref[self.tindex]

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

            #if self.tindex == len(self.testfiles)-1:
            #    print "Hello?"
            #    self.hide()

        else:
            self.trials_left -= 1
            self.trialVariable.set("Trials left:" + str(self.trials_left))

            if self.trials_left > 0:
                print >>self.logfile,str(time.clock()),"Incorrect. Trials left:" + str(self.trials_left)
                self.feedbackVariable.set(random.choice(neg))
                t3 = self.testexpr[self.tindex][3-self.trials_left]
                self.refVariable.set(t3)
                print >>self.logfile,str(time.clock()),"RE", t3

            #elif self.tindex == len(self.testfiles)-1:
            #    print "Hello?", self.tindex
            #    self.hide()

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

        print len(tfiles),len(tregions),len(tsets),len(self.clicks),len(self.success),len(self.lastclick)
        logframe = pd.DataFrame({'file':tfiles,'region':tregions,'testset':tsets,'refexp':self.testexpr,'reftype':self.testref,\
            'clicks':self.clicks,'success':self.success,'lastClick':self.lastclick,'start':self.nextclick})
        logframe.to_csv(self.logname[:-4]+"_df.csv")


        self.feedbackVariable.set("THANK YOU!")
        self.trialVariable.set("This was a good piece of work.")


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
