# reddit-r-bitcoin
data scraping and machine learning project

# summary
downloader.py pulls submissions from the r/bitcoin subreddit.
process_data.py formats the data and trains LSTM model

# requirements
praw - reddit api
pandas, re, keras
nltk - filter unnecesary words

glove.6B.100d.txt - download from https://nlp.stanford.edu/projects/glove/
nltk model: 'punkts' - can be downloaded by nltk.download() command 
