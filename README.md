# GAN-discriminator-Setback
This is an experiment to improve gan training through refresh learning

![alt text](https://github.com/stevenjohnlarsen/GAN-discriminator-refresh-learning/blob/main/algorithm_white.png)

The above depicts our training aprroch.
In short, every so often reset the discriminator to what it was some time ago.

## Running Training
To run a train of a model do the following:
* Make sure you have the folder structure set up
  * models/gan_models
  * data/celeba/<img_class_folers>
* Run main.py with the following command line arguments:
  * \# of iterations
  * \# of latent dim (use 100)
  * phi, used for weighting freature matching we used 0.9
  * setback_frequency (must be less than iterations)
  * setback_percentage (as a floating point number)
* setback_frequency and setback_percentage are optional, but either both or none must be used

The repository has some notebooks for visulazation and some training, but the majority of training for the paper was done using main.py
