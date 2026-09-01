import pandas as pd
import numpy as np
import scipy.linalg as linalg
from kaggle_install import path

csv_path = path + f"\\incom2024_delay_example_dataset.csv"
data_frame = pd.read_csv(csv_path)

def generate_transition_matrix(df):
    delay_time_series = np.array(df["label"].tolist())
    transition_dict = {}
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            transition_dict[(x,y)] = 0
    for n in range(len(delay_time_series)-1):
        transition_dict[delay_time_series[n],delay_time_series[n+1]] += 1
    first_row_total = transition_dict[-1,-1] + transition_dict[-1,0] + transition_dict[-1,1]
    second_row_total = transition_dict[0,-1] + transition_dict[0,0] + transition_dict[0,1]
    third_row_total = transition_dict[1,-1] + transition_dict[1,0] + transition_dict[1,1]
    transition_matrix = [
        [transition_dict[-1,-1]/first_row_total, transition_dict[-1,0]/first_row_total, transition_dict[-1,1]/second_row_total],
        [transition_dict[0,-1]/second_row_total, transition_dict[0,0]/second_row_total ,transition_dict[0,1]/second_row_total],
        [transition_dict[1,-1]/third_row_total, transition_dict[1,0]/third_row_total, transition_dict[1,1]/third_row_total]
    ]
    return transition_matrix

def broad_delay_analysis(df):
    transition_matrix = generate_transition_matrix(df)
    for row in transition_matrix:
        print(row)
    eigvals, left_eigvecs = linalg.eig(transition_matrix, left=True, right=False)
    stationary = left_eigvecs[:, 0].real
    stationary /= stationary.sum()
    print("Stationary distribution:", stationary)

broad_delay_analysis(data_frame)