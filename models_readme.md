# Reference for trained models



## OK Network.

**T1**
* Ok network V1 with probabilistic dataset.
* First model that worked.

**T6 (Did not work)**
* Ok network V1 with sigmoid at the output.
* Trained with correct/incorrect dataset.

**T7 (Did not work)**
* Ok network V2 with sigmoid at the output.
* Trained with probabilistic dataset.

**T8**
* Going back to the original model that worked.
* OKnetV1 with probabilistic dataset.
* No sigmoid function at the end.

**T9**
* Same as 8 but reducing the features of the kinematic encoder by 3. kin_encoder_size = 2048/3
* Kinematic embedding achieved a 37% classification accuracy


# Additional ideas
* Combine encoder-decoder net with our approach. 
* Add the gesture labels to training (It makes the approach no longer unsupervised.)