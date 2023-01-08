
[Task 01]
1)Reading the CSV files, performing basic descriptions of each attribute, ploting them, counting unique occurences etc.
Here we observe, that contain schema ACM contains less records than schema DBLP. So, intuitvely, we make an assumption that we should make the ACM fixed as a benchmark, 
and look for its matches.
Then, utilizing special python librares for each step of the following, preproccesing/normalization pipeline: lowering the strings, removing numbers from strings, 
removing special characthers from strings, performing stemming, removing stopwords, removing possible HTML tags.
After the preprocessing is done, a neccesary transformation is done, which will later help in performing blocking.
By simple observing the similiarties between 'venue' attribute values from both schemas, we see that a transforming can be done. 
The pairing between the two value sets can be done based on solemnly Levenstein similarity, by ordering the similarities, fixing and connecting the ACM venue values with the DBLP 
venue values. In such a way, that firstly a match with the highest Levenstein similarity is made. This yields in having a first pair of the two venues which are declared as the same,
the DBLP value for this pair is then removed as a potential candidate for the next matches(pairs) with the ACM venue values. Then, a match with second highest similarity is taken.
The DBLP value for this pair is then also removed as a potential candidate for the next matches. In the end, this results, in having 5 pairs, which can be placed in a dicitonary,
used to rename the venue values from one column to the venue values of the other column, and viceversa. 

2)
By having  columnn 'venue' of one schema renamed to the values of the other one, an efficent blocking scheme can be implemented  based on  having exactly the same 
values on attributes: 'venue', 'year'. This is a sort of a hybrid blocking, because, essentialy it is based on a string distance between the values(venues)-
before the renaming, combined with exact values(year). A pair (A,B) from a block, represents, a record A from ACM, and a record B from DBLP, which have the same venue, 
and year as mentioned.
3)
For each pair in the block, a scoring is performed. By being in the same block, for venue, year they get score 1. For attributes author, title, similarity based on 
jaro-winkler distance is calculated. Filtering is done, by a condition, that a pair in the block is a possible match if its overall sum of scores on author, title
is greater than 1.60. Some records from ACM can have multiple potential candidates from DBLP, even after the filtering. However, a record which has the heighest score is choosen afterall.
It is important to note here, that this pairs represent a MultiIndex made by combining indices from both tables. 

In order to perform any evaluation of our perfect matches, PerfectMapping.csv needs to be transformed into pairs. This is done by exploiting the uniqueness of both ids.
Once again, a dictionary can be created used for transforming the (idACM,idDBLP) pairs into pairs ( index of idACM from ACM, index of idDBLP from DBLP).
