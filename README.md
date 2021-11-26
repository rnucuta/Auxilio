# Auxilio

Program meant to predict the incidence levels of valley fever in Arizona based on Google Trends search data centered around the time of year. This program is trained with about 10 years of google trends data for search terms related to valley fever. Using this data, which can be found in /dumps, I trained differened machine learning models from the sklearn python library on the data to predict the incidence rates of valley fever in the current week. This output is classified into low (<30 patients), medium (30-70 patients), and high (>70 patients) incidence.

To run the program, you need linux on your computer, or you need to download the [ubuntu linux subsytem](https://www.microsoft.com/store/productId/9NBLGGH4MSV6) for windows. Instructions for how to do so are [here](https://docs.microsoft.com/en-us/windows/wsl/install). Once this is installed, run the python virtual environment in the ubuntu subsytem by cding into the projects repository, and running the command `source penv1/bin/activate`. This enters you into the python virtual environment. From here, to run inference, you simply run the inference.py file with its default command, which can be found in inference.py. If you want to make your own model, run each of the programs in the following order using the default command comments found in each python file: data_processor.py, training.py, and inference.py.
