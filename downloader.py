# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 19:49:14 2017

this script downlads posts in a subreddit
@author: Lorand
"""

#link for getting started with reddit api
#https://github.com/reddit/reddit/wiki/OAuth2

#link for praw
#https://praw.readthedocs.io/en/latest/getting_started/quick_start.html

#link for rules of using reddit API
#https://github.com/reddit/reddit/wiki/API




import praw #reddit api wrapper
from datetime import datetime
import pandas as pd
import re
import numpy as np

import string
import time
import csv

#connect to reddit
reddit = praw.Reddit(client_id='client_id',
                     client_secret='client_secret',
                     user_agent='download-comments v1.0.20 (by /u/whitewalker111)',
                     username='username',
                     password='username_password')

subreddit = reddit.subreddit('bitcoin')

print(subreddit.display_name)  # Output: redditdev

maxlimit = 100

#options for sorting the submissions before iterating through:
#controversial,gilded,hot,new,rising,top

sub_list = []
comment_list = []
###############################################################
#todo useful parts of a comment
# 1 - upvote/downvote ratio( or subtraction/maxUPvotes ) -
#     maybe weight by the total amount of votes
# 2 - actual content

# 3 - who posted it maybe??? - for later
# 3.1 - who is trustworthy, -> his comments are usually low DOWNvote/UPvote

# update database, download new submision and compare to the saved ones
# if already exsist -> delete submission   
###############################################################
print('load submissions and get comments')
for submission in subreddit.hot(limit=maxlimit):
    sub_list.append(submission)
    #link for extracting comments from submission
    #http://praw.readthedocs.io/en/latest/tutorials/comments.html
   
    queue = submission.comments[:]  # Seed with top-level
    try:
        while queue:
            comment = queue.pop(0)
            comment_list.append(comment)
            queue.extend(comment.replies)
    except AttributeError:
        pass
    
# csv file content
# submission :id,score,selftext + title
# comment :id,score,body
# for later -> author, created_utc

# the attributes of submission and comment and etc..
# link: https://github.com/reddit/reddit/wiki/JSON
print(' remove unnecesary characters')
def clean(raw):
    #todo remove links
    res = re.sub("[^a-zA-Z0-9]"," ", raw)
    return res
################################################################
#for creating csv
sub_data = []
for i, item in enumerate(sub_list):
    #remove unnecesary caracters 
    sub_text = clean( sub_list[i].selftext ) + clean( sub_list[i].title )
    #create list to save to csv
    tmp = [sub_list[i].id,sub_list[i].score,sub_text]
    sub_data.append(tmp)
    
com_data = []
for i, item in enumerate(comment_list):
    try:
        #remove unnecesary caracters
        comment_list[i].body = clean( comment_list[i].body )
        #create list to save to csv
        tmp = [comment_list[i].id,comment_list[i].score,comment_list[i].body]
        com_data.append(tmp)
    except AttributeError:
        pass
################################################################

print('submissions size = {}'.format( len(sub_data)))
print('comments size = {}'.format(    len(com_data)))

print('wrinting data')
col = ['id', 'score', 'text']
df_sub = pd.DataFrame(sub_data, columns = col)
df_sub.to_csv("submission.csv", index = False)


df_com = pd.DataFrame(com_data, columns = col)
df_com.to_csv("commment.csv", index = False)



    