import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

congress = 117 # Temporary variable, we'll probably just hardcode 117th Congress in for a specific question
name = '[rep_name]'

data = np.genfromtxt(f'{congress}embedding.csv', delimiter=',') # This is definitely the best clustering result
colors = np.genfromtxt(f'{congress}colors.csv', delimiter=',')

if congress == 116:
    df = pd.read_csv('vote_data_116/h743.csv')
if congress == 117:
    df = pd.read_csv('vote_data_117/h384.csv')
if congress == 118:
    df = pd.read_csv('vote_data_118/h1.csv')
df.drop(0, inplace = True)
df['name'] = df['name'].str.removeprefix('Rep. ').str.replace(r' \[.*\]', '', regex = True)
name_id = dict(zip(df['name'], df.index))

fig, ax = plt.subplots()
ax.scatter(data[:,0], data[:,1], c = colors)
plt.scatter(data[name_id[name], 0], data[name_id[name], 1], c = 'green', marker = 'X', s = 100)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_title(f'{congress}th Congress - House of Representatives Voting Patterns')

st.pyplot(fig)