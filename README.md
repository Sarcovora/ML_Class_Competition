This branch contains our CatBoost and LightBGM models. The data is cleaned and features are engineered. However, for the CatBoost model, we skip the one-hot encoding step, as it deals with categorical data natively.

Some things that produced increasingly good results as they were implemented over different iterations of this model:

- Making sure to have models optimize for class balance and letting that be our scoring function
- Tuning LightBGM parameters using GridSearch
- Using feature importance to select what features to engineer/drop
- Rare encoding rare features with less than some minFreq (settled on 1%)

other things that were tried but did not work as well:
- Separating breeds / colors into Breed 1 and Breed 2 and Color 1 and Color 2: Did not meaningfully change validation accuracy, and not enough attempts to see how it would do in contest.
