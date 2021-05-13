# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from PIL import Image
import time
import numpy as np
import matplotlib.pyplot as plt
import ast
from six.moves import urllib
from pathlib import Path

# variables
rf_images = ["1yr.png", "1yr.png", "2yr.png", "3yr.png", "4yr.png", "5yr.png",
            "7yr.png", "7yr.png", "10yr.png", "10yr.png", "10yr.png"]

lr_images = ["lr_1yr.png", "lr_1yr.png", "lr_2yr.png", "lr_3yr.png", "lr_4yr.png", "lr_5yr.png",
             "lr_7yr.png", "lr_7yr.png", "lr_10yr.png", "lr_10yr.png", "lr_10yr.png"]

svm_images = ["svm_1yr.png", "svm_1yr.png", "svm_2yr.png", "svm_3yr.png", "svm_4yr.png", "svm_5yr.png",
              "svm_7yr.png", "svm_7yr.png", "svm_10yr.png", "svm_10yr.png", "svm_10yr.png"]

km_images = ["km_1yr.png", "km_1yr.png", "km_2yr.png", "km_3yr.png", "km_4yr.png", "km_5yr.png",
             "km_7yr.png", "km_7yr.png", "km_10yr.png", "km_10yr.png", "km_10yr.png"]

about_us_text = 'We are a team of developers with a great passion for data science and problem-solving.\
             In 2020, our research team discovered the topic of Publication Prediction Performance \
             while exploring subjects for our capstone project. Our efforts are focused on using information \
             about peer-reviewed publications to build a predictive model which can calculate the rate of citation \
             of a given paper. We understand that technology forecasting plays a critical role in the operation of \
             many organizations. By utilizing the principles of machine learning and data science, we were able to \
             design and develop a program to predict the number of citations a paper receives in the years following \
             publication. Our team is excited to introduce a simplified form of citation prediction through this application.'

results_text = 'Our study successfully addresses the problem of Prediction Publication Performance by \
             analyzing multiple state of the art solutions regarding citation prediction. \
             The evaluation was conducted using MSE, RMSE, and MAE. As seen in the table, \
             the value for MAE and RMSE increased over the period of citation. Meaning, the \
             average value of residuals’ magnitude and the square root of the residuals’ \
             variance increased as we approached the 10 year citation prediction. Hence, \
             the model accuracy has an inverse relation in respect to the time passed. As seen \
             in the percentage histogram of the yearly citations (Figure 4) we observe that the \
             value distribution of the train set, test set, and the model prediction values are \
             fairly similar to one another. This proves that the predictive model can successfully \
             track the data trends and is not overfitted on the training set. The random forest decision \
             tree shows the most promising results during the first year of citation prediction \
             (with MAE of 1.1689, MSE of 43.3886, and RMSE of 6.5870). It was observed that the author \
             and venue features show higher correlation with the citation than the paper features. The \
             results of the study implies that non-linear regression models perform significantly better \
             than linear regression models for the citation prediction problem. The ensemble methods improved \
             the model performance overall by combining several simpler machine learning models and decreasing \
             their bias and variance.'

demo_text_1 = 'In the rise of the current rate of publication, creating and identifying influential papers are \
             challenging tasks for researchers. The objective of this project is to predict the impact of an \
             academic paper in future using the citations metric. The research team performs an extensive study \
             into the current methodologies for this machine learning problem. Consequently, the project arrives \
             at multiple ensemble learning algorithms to perform the task.'

demo_text_2 = 'This study utilizes the famous [Aminer dataset](https://www.aminer.org/citation) for training and testing \
             purposes. We examine 10 independent features related to papers, authors, and venues to discover the highly \
             cited papers’ patterns and predict their influence within one to ten years of publication. The model’s training \
             set consists of over 1 million academic papers while the testing set covers over 200,000 papers. More details \
             are provided in the [report](https://drive.google.com/file/d/1zMH1FKa_1LqWtWLVhxDYSeCywIKeLKvl/view?usp=sharing) and \
             the [github](https://github.com/erfankashani/citation_prediction).'

main_path = Path(__file__).parents[1]

demo_text_3 = 'To start the demo please choose a paper from the dropdown field:'

st.title('Citation Prediction')

# chache the time consuming tasks in the beginging
@st.cache()
def get_demo_data():
    demo_data = pd.read_csv(str(main_path / 'Results/demo_data_frame.csv'))
    return demo_data

# Edits paper dataframe for table display
def get_paper_table(paper):
    paper_info = paper[["authors", "year", "venue", "fos", "citation"]].copy()
    paper_info['authors'] = ast.literal_eval(paper_info['authors'])
    paper_info['venue'] = ast.literal_eval(paper_info['venue'])
    paper_info['fos'] = ast.literal_eval(paper_info['fos'])
    
    paper_info['authors'] = ', '.join([i['name'] for i in paper_info['authors']])
    paper_info['venue'] = paper_info['venue']['raw']
    paper_info['fos'] = ', '.join([i['name'] for i in paper_info['fos']])
    
    paper_info = paper_info.rename(index={'authors': 'Authors',
                                          'venue': 'Venue', 
                                          'fos': 'Fields of Study', 
                                          'year':'Year', 
                                          'citation':'Citation'})
    return paper_info

# Edits features dataframe for table display
def get_feature_table(paper):
    paper_info = paper[['diversity', 'venue_rank', 'venue_MPI',
                        'venue_TPI', 'productivity', 'H_index', 
                        'author_rank', 'author_MPI','author_TPI', 
                        'versatility']]
    
    paper_info = paper_info.rename(index={'diversity': 'Diversity', 
                                          'venue_rank': 'Venue Rank', 
                                          'venue_MPI': 'Venue MPI', 
                                          'venue_TPI': 'Venue TPI',
                                          'productivity': 'Productivity', 
                                          'H_index': 'H Index',
                                          'versatility': 'Versatility', 
                                          'author_rank': 'Author Rank', 
                                          'author_MPI': 'Author MPI', 
                                          'author_TPI': 'Author TPI'})
    return paper_info

# Edits prediction dataframe for table display
def get_prediction_table(paper):
    paper_info = pd.DataFrame(columns=[ 'index','Prediction','Ground Truth'])
    paper_info['Prediction'] = paper[['rf_predict_1yr', 'rf_predict_2yr', 'rf_predict_3yr', 
                                   'rf_predict_4yr', 'rf_predict_5yr', 'rf_predict_7yr', 'rf_predict_10yr']]
    
    paper_info['Ground Truth'] = paper[['citation_1yr', 'citation_2yr', 'citation_3yr', 
                                    'citation_4yr', 'citation_5yr', 'citation_7yr', 'citation_10yr']].values

    paper_info['index']= ['Citation after 1 year',
                          'Citation after 2 years',
                          'Citation after 3 years', 
                          'Citation after 4 years',
                          'Citation after 5 years',
                          'Citation after 7 years',
                          'Citation after 10 years']
    paper_info = paper_info.set_index("index")
    return paper_info

# Renders Demo page
def show_demo():
    st.write(demo_text_1)
    st.write(demo_text_2)
    st.write(demo_text_3)

    demo_df = get_demo_data()
    demo_df = demo_df.set_index("title")
    
    papers = st.selectbox(
        "Choose paper", list(demo_df.index)
    )

    # Progress bar
    if (papers):     
        # Add a placeholder
        latest_iteration = st.empty()
        bar = st.progress(0)
        for i in range(100):
          # Update the progress bar with each iteration.
          latest_iteration.text(f'Completion {i+1}%')
          bar.progress(i + 1)
          time.sleep(0.01)
    
    # show the raw data
    st.write("\n The paper's information:")
    paper = demo_df.loc[papers]
    paper_info_df = get_paper_table(paper)
    st.write(paper_info_df)
    
    # show the features
    st.write("\n Calculated paper's features:")
    feature_info_df = get_feature_table(paper)
    st.write(feature_info_df)
    
    # show prediction (table and graph)
    st.write("\n Predictions:")
    prediction_info_df = get_prediction_table(paper)
    st.write(prediction_info_df)

    # graph
    st.write("\n Comparison graph:")
    xVals = ['1 year','2 years','3 years','4 years','5 years','7 years','10 years']
    yVal1 = paper[['rf_predict_1yr', 'rf_predict_2yr', 'rf_predict_3yr', 'rf_predict_4yr', 'rf_predict_5yr', 'rf_predict_7yr', 'rf_predict_10yr']].copy().to_numpy()
    yVal2 = paper[['citation_1yr', 'citation_2yr', 'citation_3yr', 'citation_4yr', 'citation_5yr', 'citation_7yr', 'citation_10yr']].copy().to_numpy()
    fig, ax = plt.subplots()
    ax.fill_between(xVals, 0, np.transpose(yVal1).tolist(), color="darkkhaki", alpha=0.5)
    ax.fill_between(xVals, 0, np.transpose(yVal2).tolist(), color="teal", alpha=0.5)
    ax.legend(['Predicted Values','True Values'])
    ax.margins(0.1)
    st.pyplot(fig)

# Renders result page
def show_results():
    st.write(results_text)

    # MAE graph
    image_mae = Image.open(str(main_path / 'Results/image/mae.png'))
    st.image(image_mae, caption='Figure 1: Mean Absolute Error')

    # RMSE graph
    image_rmse = Image.open(str(main_path / 'Results/image/rmse.png'))
    st.image(image_rmse, caption='Figure 2: Root Mean Square Error')

    # R Square praph
    image_r_square = Image.open(str(main_path / 'Results/image/r_square.png'))
    st.image(image_r_square, caption='Figure 3: R Square')

    # Random Forest slider graph
    st.write("Random Forest model performance:")
    year_rf = st.slider('Number of years after publication?',1,10)    
    year_image = 'Results/image/' + rf_images[year_rf]
    rf_image = Image.open(str(main_path / year_image))
    st.image(rf_image, caption='Figure 4: Random Forest Citation Frequency Histogram', width=600)

    # Random Forest performance table
    st.write("The following are the performance matrix for Random Forest Regression model:")
    performance_df = pd.DataFrame(np.array([[1.168999, 1.89990, 2.66408, 3.43690, 4.19492, 5.62216, 6.94813],
                                 [43.38860, 71.76869, 124.98958, 216.97061, 356.03481, 86.33918, 1421.48745],
                                 [6.58700, 8.47164, 11.17987, 14.72992, 18.86888, 28.04174, 37.70262],
                                 [0.16119, 0.22493, 0.27972, 0.303861, 0.30706, 0.29510, 0.22525]]),
                       columns=['1 year', '2 year', '3 year', '4 year', '5 year', '7 year', '10 year'])
    performance_df['index']= ["MAE","MSE","RMSE", "R Square"]
    performance_df = performance_df.set_index("index")
    st.table(performance_df)

    # Benchmark Models slider graphs
    st.write("Benchmark model performance:")
    year_lr = st.slider('Number of years after publication?',0,10)
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        lr_year_image = 'Results/image/lr/' + lr_images[year_lr]
        lr_image = Image.open(str(main_path / lr_year_image))
        st.image(lr_image, caption='Figure 5: LR Citation Frequency Histogram')
    with col2:
        svm_year_image = 'Results/image/svm/' + svm_images[year_lr]
        svm_image = Image.open(str(main_path / svm_year_image))
        st.image(svm_image, caption='Figure 6: SVM Citation Frequency Histogram')
    with col3:
        km_year_image = 'Results/image/kmeans/' + km_images[year_lr]
        km_image = Image.open(str(main_path / km_year_image))
        st.image(km_image, caption='Figure 7: K-Means Citation Frequency Histogram')

# Renders about us page
def show_about_us():
    st.write(about_us_text)

    st.subheader("\nOur team:")
    col1, col2 = st.beta_columns(2)
    with col1:
        st.subheader("Fatima Tafazzoli Shadpour")
        st.image(str(main_path / 'Results/image/fatima.jpeg'),use_column_width='always')
        st.subheader("Erfan Kashani")
        st.image(str(main_path / 'Results/image/erfan.jpeg'),use_column_width='always')
    with col2:
        st.subheader("Lubaba Tasnim")
        st.image(str(main_path / 'Results/image/Lubaba.jpeg'),use_column_width='always')
        st.subheader("Muhammad Karim")
        st.image(str(main_path / 'Results/image/hamza_1.png'),use_column_width='always')

try:
    # Navigation Menu
    menu_choice = st.sidebar.selectbox(
        "Navigate to:", ["Demo", "Results", "About us"], index=0
    )
    if menu_choice == "Demo":
        st.subheader("The Demo:")
        show_demo()
    elif menu_choice == "Results":
        st.subheader("The Results:")
        show_results()
    else:
        st.subheader("About us:")
        show_about_us()

except urllib.error.URLError as e:
    st.error(
        """
        **In Case of Errors.**
        error: %s
        """
        % e.reason
    )
