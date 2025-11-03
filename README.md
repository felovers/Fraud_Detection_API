# Fraud_Detection_API
Fraud Detection System as an API

In this repository, there are the files for the API, its test and the gui, which is the system (should be put into an executable, that is too big to be uploaded here).

The file app.py is the code for the API, it would receive the data from the creditcard.csv (which can be found here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), or any other data that has the same pattern, and send it to the model. The model would process the data and return its predictions.

The software file (gui.py) would open a screen that you can select the csv file you wish to use, then send to the API (that executes like demonstrated above). Using the predictions returned, it will show metrics of the model, evaluating it, and show 2 buttons: 1 to show the Confusion Matrix and 1 to issue a report on the detected frauds.

The test_api.py file is a simple test for the API, the GUI file contains a better version of the code, optimized for usage.
