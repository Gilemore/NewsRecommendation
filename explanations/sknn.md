# Session-based KNN

## The basic idea

For one session, given the previous item (articles or categories sequence), we have all the sessions that contain this item (neighbors) and calculate their similarity with our current session. Then from neighbors, we can get the scores for each item. And we rank them to get the top 20, use Recall@20 or MRR@20 to evaluate this method.

## Details

### Fit data

We train the data so we get several dictionaries.   

```python
session_item_map={1:[1,2,3],2:[56,23]....}
item_session_map={1:[14,34,25,234,536,2877,...],2:[...],...}
session_time:{1:2489236,2:328953...}
```

### Predict 

given session id and one previous item, we try to predict the next item in this session.

+ update the dictionaries
+ find neighbor sessions that also contains this previous item
+ calculate their similarity (cosine) (focus on orientation not magnitude)
+ get the score of every item
+ rank the item and evaluate this prediction

### Evaluate

+ Recall@n: wether the true prediction is in the top n prediction ranking
+ MRR@n: inverse value of the rank of the true prediction

## What I have tried 

At first I thought we can add some features into the model: calculating the sessions cosine similarity in order to improve performance.

# VMSKNN

Vector Multiplication session-based KNN:

in calculating similarity of sessions, we add a decaying function so that time closer session will have a higher similarity and farther session will have a lower similarity. 