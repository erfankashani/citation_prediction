# Citation Prediction

In the rise of the current rate of publication, creating and identifying influential papers are challenging tasks for researchers. The objective of this project is to predict the impact of an academic paper in future using the citations metric. The research team performs an extensive study into the current methodologies for this machine learning problem. Consequently, the project arrives at multiple ensemble learning algorithms to perform the task.

This study utilizes the famous [Aminer dataset](https://www.aminer.org/citation) for training and testing  purposes. We examine 10 independent features related to papers, authors, and venues to discover the highly cited papers’ patterns and predict their influence within one to ten years of publication. The model’s training set consists of over 1 million academic papers while the testing set covers over 200,000 papers. More details are provided in the [report](prediction_publication_performance.pdf).

You can view the website [here](www.google.com).

#### To start the front-end locally follow the instructions

install streamlit

```
cd /path/to/project
pip3 install -r requirements.txt
```

run the front end

```
cd Code
streamlit run visualize.py
```

Then open the link provided on the commandline