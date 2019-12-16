# IKNN

## Overview

Item-to-item kin: return the most similar items to the last read article using the cosine similarity between their vectors of co-occurrence with other items within sessions. 

$s_{i,j}=\sum_{s}I\{(s,i)\in D  (s,j)\in D\}$

$(supp_i+\lambda)^{\alpha}(supp_j+\lambda)^{1-\alpha}$

## Details

## Fit

get the co-occurence matrix. For each item, we calculate the similarity (number of times) that this item and each other item happens in one session.

## Predict

for the next item, we look up in the co-occurence matrix and get the similarity of each other items with the item. We rank items and make our prediction.