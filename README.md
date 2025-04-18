This branch contains the best model we were able to get. It is a bagged MLP (10 estimators seemed to work the best) with each MLP being 3 layers: input, hidden, and output layers.

Some things that produced increasingly good results as they were implemented over different iterations of this model:
- bagging vs single model
- using embeddings rather than one hot encoding the breed and color as these are high cardinality categorical features
- tuning the embedding sizes
- tuning the number of layers and neurons in the MLP
- setting the cross entropy loss to incorporate class balancing, aka incorporating the accuracy that we would be evaluated on
    - this ended up leading to very accurate validation accuracies

other things that were tried but did not work as well:
- adaboosting: hypothesized that this would work well but it got easily outperformed by the bagging strategy even without much tuning
