# AML Model Deployment Template

The following files are intended to provide a template within which a data science user can place their model script
for training within AML.

## Target User

The intended end-user profile for this code is someone who may be competent with pandas/scikit-learn,
but may not be comfortable modifying code outside of this framework.

We have taken the boilerplate AML code and abstracted this away into discrete non-user files (a `main.py` script and `config.json` / `user.json` files, all not shown here).

There are specific user-modifiable scripts linked to the model training process:
* `src/preprocess.py` - a script that accepts a pandas DataFrame as input, and returns 2/4 numpy arrays (X_train, y_train, and option for X_test/y_test)
* `src/model.py` - a script containing a function that accepts X/y numpy arrays (train and test) as input, and returns a trained model along with metrics.

There are additional scripts which help to package up the above two scripts and communicate these to the AML interfacing scripts.
* `src/create_data.py` - creates the pandas DataFrame, which will get processed by the `preprocess.py` script, and then save the resultant training/testing arrays
* `src/data_versioning.py` - currently empty placeholder file, representing a potential script that will pull in EMAP data as a 'version'
* `src/train.py` - loads the saved numpy train/test arrays, feeds them into the `src/model.py` script, and logs the metrics.
