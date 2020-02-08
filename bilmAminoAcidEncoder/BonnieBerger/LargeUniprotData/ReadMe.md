
We keep labels with frequencies 3 or higher.
Test data contains at least 2 observations with label at frequency 5 or below, and then train data contains the other observations having these same labels. This strategy allows us to train on some sequences with rare labels, and then evaluate on these same rare labels. A pure zeroshot approach will not work well yet.
The rest of the observations are then randomly divided into train/dev/test data.
