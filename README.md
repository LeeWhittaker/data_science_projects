# data_science_projects

>This repository contains various Jupyter notebooks that I have worked on while trying to learn about machine learning and data cleaning in the context of data science.

There are currently two Python scripts that are used in the notebooks that can be installed uisng the ```setup.py``` script. These are ```simple_neural_net.py``` and ```tensorflow_neural_net.py```, which are contained in the ```python``` folder.

To install these, you can run
```
python setup.py install --user
```

They are used in the ```simple_nn_with_mushrooms.pynb``` and ```tensorflow_with_medical_insurance.ipynb``` within the ```notebooks/my_projects``` folder.

```simple_neural_net.py``` is a set of Python functions that I wrote while doing the [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning) Coursera course, which is part of the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning).

```tensorflow_neural_net.py``` is an ongoing attempt to recast and improve ```simple_neural_net.py``` in Tensorflow (with the Keras API), using methods introduced to me while doing the [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects?specialization=deep-learning) course, which is also part of the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning).


## Folder contents

The ```data``` folder contains all of the data used within the notebooks. Links to where the data was obtained are given in the notebooks.

The ```notebooks``` folder contains the main projects that I have been working on.

In ```notebooks/example_notebooks``` there are a coupe of notebooks that I downloaded from Kaggle to help get me started. I have recast them, with some modifictions and/or extensions in ```notebooks/initial_notebooks```.

In ```notebooks/my_projects``` there are two projects from Kaggle. One is using the mushrooms dataset to predict whether a mushroom is edible or poisoness and the other is looking at predicting medical insurance costs. Each of these have been looked at twice, once using standard machine learning techniques, sich as decision trees and linear regression, and a second time using the neural network codes discussed above.

In ```notebooks/data_cleaning``` there are two projects where I have started with unclean data. The first project is using TV program information downloaded as libsvc sparse matrix, which I then recast into a dataframe and use to predict whether we are looking at a commercial or not. The second is file of excel spreadsheets giving Covid-19 deaths in the UK, with age, region, and sex information. This is used to invesigate the whether the correct amount of deaths has been attributed to Covid-19, whether the death rate is higher for males or females, and whether one region is impacted higher than another.
