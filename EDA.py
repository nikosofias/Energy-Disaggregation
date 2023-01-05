# %% [markdown]
# ## Energy Disaggregation (Exaplanatory Data Analysis)

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

# %% [markdown]
# ### Import datasets

# %%
os.getcwd()

# %%
hot_plate = pd.read_csv(r'/Users/nikossofias/Downloads/Αντιστασιακών φορτία/Energy-Disaggregation/90642-3/Hotplate.csv')
water_heater = pd.read_csv(r'/Users/nikossofias/Downloads/Αντιστασιακών φορτία/Energy-Disaggregation/90642-3/Total_house__Water_heater.csv')

# %%
### Preprocessing DATES

def combine_date_time(x, y):
        return datetime.combine(x, y)

def datetime_preprocessing(df):
    df['Time (UTC)'] = df['Time (UTC)'].apply(lambda x: datetime.strptime(x,'%H:%M:%S:%f').time())
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['date_time'] = list(map(lambda x, y: combine_date_time(x, y), df['Date'].values.tolist(), df['Time (UTC)'].values.tolist()))
    df = df.drop(['Date', 'Time (UTC)'], axis=1)
    return df

hot_plate = datetime_preprocessing(hot_plate)
water_heater = datetime_preprocessing(water_heater)

# %%
water_heater = water_heater[['date_time','Active Power L1 (W)', 'Reactive Power L1 (Var)', 'Active Power L3 (W)']]
hot_plate = hot_plate[['date_time','Active Power L1 (W)']]

# %%
water_heater.info()

# %%
hot_plate.info()

# %%
water_heater.plot(x='date_time',subplots=True,figsize=(20,6))

# %%
from turtle import color, width
from plotly import graph_objects as go

fig2 = go.Figure(
    data=[
        go.Line(
            name="Θερμοσύφωνας [W]",
            x=water_heater.date_time,
            y=water_heater['Active Power L3 (W)'],
            legendgroup=1,line=dict(color="black")
        ),
        go.Line(
            name="Ενεργός Ισχύς σπιτιού [W]",
            x=water_heater.date_time,
            y=water_heater['Active Power L1 (W)'],
            legendgroup=2,line=dict(color="red")
        ),
        go.Line(
            name="Άεργος Ισχύς σπιτιού [Var]",
            x=water_heater.date_time,
            y=water_heater['Reactive Power L1 (Var)'],
            legendgroup=3,line=dict(color="blue")
        ),
        go.Line(
            name="Ενεργός Ισχύς Μάτια Κουζίνας [W]",
            x=hot_plate.date_time,
            y=hot_plate['Active Power L1 (W)'],
            legendgroup=4,line=dict(color="green")
        ),
    ],
    layout=go.Layout(
        title="Κατανάλωση Σπιτιού & Θερμοσύφωνα",
        yaxis_title="Power [W]"
    )
)

fig2.update_layout(
    margin=dict(l=20, r=20, t=30, b=10)
)
fig2.show()

# %% [markdown]
# ### Ενεργός & Αεργος Ισχύς EDA

# %% [markdown]
# Πανω στην Ενεργο Ισχύ & αεργο του σπιτιού θα εφαρμοστούν οι κανονες

# %%
water_heater['Active Power L1 (W)'].plot(title='Total house Active Power [W]')
plt.grid()
plt.figure()
water_heater['Active Power L1 (W)'].hist(bins=100)
plt.title('Histogram Total house Active Power [W]')
print(water_heater['Active Power L1 (W)'].describe())

# %%
water_heater['Reactive Power L1 (Var)'].plot(title='Total house Reactive Power (Var)')
plt.grid()
plt.figure()
water_heater['Reactive Power L1 (Var)'].hist(bins=100)
plt.title('Histogram Total house Reactive Power (Var)')
print(water_heater['Reactive Power L1 (Var)'].describe())

# %%
fig2 = go.Figure(
    data=[
        go.Line(
            name="Ενεργός Ισχύς σπιτιού [W]",
            x=water_heater.date_time,
            y=water_heater['Active Power L1 (W)'],
            legendgroup=1,line=dict(color="black")
        ),
        go.Line(
            name="1η διαφορά Ενεργός Ισχύς σπιτιού [W]",
            x=water_heater.date_time,
            y=water_heater['Active Power L1 (W)'].diff(),
            legendgroup=2,line=dict(color="red")
        ),
        go.Line(
            name="Άεργος Ισχύς σπιτιού [Var]",
            x=water_heater.date_time,
            y=water_heater['Reactive Power L1 (Var)'],
            legendgroup=3,line=dict(color="blue")
        ),
        go.Line(
            name="1η διαφορά Άεργος Ισχύς σπιτιού [Var]",
            x=water_heater.date_time,
            y=water_heater['Reactive Power L1 (Var)'].diff(),
            legendgroup=4,line=dict(color="green")
        ),
    ],
    layout=go.Layout(
        title="Κατανάλωση Σπιτιού",
        yaxis_title="Power [W]"
    )
)

fig2.update_layout(
    margin=dict(l=20, r=20, t=30, b=10)
)
fig2.show()

# %%
# water_heater['Active Power L1 (W)'].rolling(100).apply(max).dropna().reset_index(drop=True).plot()

# %% [markdown]
# ## Κανόνας: Απότομες Αλλαγές Active & Reactive

# %%
ap_diff = water_heater['Active Power L1 (W)'].diff()
rp_diff = water_heater['Reactive Power L1 (Var)'].diff()

# %%
ap_diff.hist(bins=100)

# %%
ap_diff_std = ap_diff.describe()['std']
ap_diff.describe()

# %% [markdown]
# ### Φιλτράρουμε το σήμα της παραγώγου της:
# - Ενεργού Ισχύος με τυπική απόκλιση των 3σ
# - Άεργου Ισχύος με τυπική απόκλιση των 2σ

# %%
# ap_diff.plot()
# ap_diff.where((ap_diff>3*ap_diff_std) | (ap_diff<-3*ap_diff_std)).plot()#.dropna().index.tolist()

# plt.figure()

# rp_diff.plot()
# rp_diff.where((rp_diff>2*rp_diff.describe()['std']) | (rp_diff<-2*rp_diff.describe()['std'])).plot()#.dropna().index.tolist()

# %% [markdown]
# ### Οπτικοποιούμε τα αποτελέσματα

# %%
fig2 = go.Figure(
    data=[
        go.Line(
            name="Ενεργός Ισχύς σπιτιού [W]",
            x=water_heater.date_time,
            y=water_heater['Active Power L1 (W)'],
            legendgroup=1,line=dict(color="black")
        ),
        # go.Line(
        #     name="1η διαφορά Ενεργός Ισχύς σπιτιού [W]",
        #     x=water_heater.date_time,
        #     y=water_heater['Active Power L1 (W)'].diff(),
        #     legendgroup=2,line=dict(color="red")
        # ),
        go.Line(
            name="Changing Points in Active Power",
            x=water_heater.date_time,
            y=ap_diff.where((ap_diff>3*ap_diff.describe()['std']) | (ap_diff<-3*ap_diff.describe()['std'])),
            legendgroup=3,line=dict(color="blue")
        ),
        go.Line(
            name="Changing Points in Rective Power",
            x=water_heater.date_time,
            y=rp_diff.where((rp_diff>2*rp_diff.describe()['std']) | (rp_diff<-2*rp_diff.describe()['std'])),
            legendgroup=4,line=dict(color="green")
        ),
    ],
    layout=go.Layout(
        title="Κατανάλωση Σπιτιού",
        yaxis_title="Power [W]"
    )
)

fig2.update_layout(
    margin=dict(l=20, r=20, t=30, b=10)
)
fig2.show()

# %%
ap_interest_points = ap_diff.where((ap_diff>3*ap_diff.describe()['std']) | (ap_diff<-3*ap_diff.describe()['std'])).dropna().index.tolist()
rp_interest_points = rp_diff.where((rp_diff>2*rp_diff.describe()['std']) | (rp_diff<-2*rp_diff.describe()['std'])).dropna().index.tolist()

# %% [markdown]
# ## Κανόνας: ~Σταθερή Κατανάλωση εν λειτουργία φορτίου

# %%
len(ap_interest_points)

# %%
ap_interest_points_stds = []
idxs_ap_interest_points_stds = []
for idx in ap_interest_points:
    # print(idx)
    ap_interest_points_stds.append(water_heater.loc[idx+1:idx+5,'Active Power L1 (W)'].min())
    idxs_ap_interest_points_stds.append(idx)

rp_interest_points_stds = []
idxs_rp_interest_points_stds = []
for idx in rp_interest_points:
    rp_interest_points_stds.append(water_heater.loc[idx+1:idx+5,'Reactive Power L1 (Var)'].min())
    idxs_rp_interest_points_stds.append(idx)

# %%
water_heater.loc[idxs_ap_interest_points_stds[100]:idxs_ap_interest_points_stds[100]+10,'Active Power L1 (W)'].plot()

# %%
plt.plot(ap_interest_points_stds)

# %%
plt.plot(rp_interest_points_stds)
pd.Series(rp_interest_points_stds).where(pd.Series(rp_interest_points_stds)<-100).plot()

# %%
rp_final_idxs = []
for special_idx in pd.Series(rp_interest_points_stds).where(pd.Series(rp_interest_points_stds)<-100).dropna().index.tolist():
    rp_final_idxs.append(idxs_rp_interest_points_stds[special_idx])

# %%
special_rp_idxs = np.zeros(len(water_heater))
special_rp_idxs = pd.Series(special_rp_idxs).replace(0,np.nan)
special_rp_idxs[rp_final_idxs] = water_heater.loc[rp_final_idxs,'Reactive Power L1 (Var)'].values

# %%
special_rp_idxs.dropna()

# %% [markdown]
# ### Οπτικοποιούμε τα αποτελέσματα

# %%
fig2 = go.Figure(
    data=[
        go.Line(
            name="Ενεργός Ισχύς σπιτιού [W]",
            x=water_heater.date_time,
            y=water_heater['Active Power L1 (W)'],
            legendgroup=1,line=dict(color="black")
        ),
        go.Scatter(
            name="Τελικά Ενδιαφέροντα σημεια Άεργος Ισχύς",
            x=water_heater.date_time,
            y=special_rp_idxs,
            legendgroup=2,line=dict(color="red")
        ),
        go.Line(
            name="Changing Points in Active Power",
            x=water_heater.date_time,
            y=ap_diff.where((ap_diff>3*ap_diff.describe()['std']) | (ap_diff<-3*ap_diff.describe()['std'])),
            legendgroup=3,line=dict(color="blue")
        ),
        go.Line(
            name="Changing Points in Rective Power",
            x=water_heater.date_time,
            y=rp_diff.where((rp_diff>2*rp_diff.describe()['std']) | (rp_diff<-2*rp_diff.describe()['std'])),
            legendgroup=4,line=dict(color="green")
        ),
    ],
    layout=go.Layout(
        title="Κατανάλωση Σπιτιού",
        yaxis_title="Power [W]"
    )
)

fig2.update_layout(
    margin=dict(l=20, r=20, t=30, b=10)
)
fig2.show()

# %%
water_heater['Active Power L1 (W)'].plot()
special_rp_idxs.plot()

# %%



