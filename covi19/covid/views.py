from traitlets.traitlets import default
from .models import *
from django.shortcuts import render
from urllib.request import urlopen 
import json 
import numpy
from datetime import date
import datetime
import requests
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# importing folium and plotly
import folium
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots 
from plotly.utils import PlotlyJSONEncoder

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima

# importing mysql
import mysql.connector

# importing seaborn
# import seaborn as sns
from plotly.offline import plot
# importing math and datetime function
import math
# from datetime import datetime,date,timedelta
from django.db import connection
# imporing scikit learn and joblib to save model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_squared_log_error
import joblib
from sklearn.model_selection import train_test_split
import requests
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')
cnf = '#393e46'
dth = '#ff2e63'
rec = '#21bf73'
act = '#fe9801'

# Create your views here.
def home(request):
    today = date.today()
    yesterday = today - timedelta(days = 1)
    url = ('https://data.covid19india.org/v4/min/data-{}.min.json'.format(yesterday))
    response = requests.get(url)
    if response:
        res = response.json()
        for key,val in res.items():
            states = key
            try:
                # print(element['total'])
                if val['total']!=None:
                    try:
                        total_confirmed=val['total']['confirmed']
                        print(val['total']['confirmed'])
                    except:
                        pass
            except:
                pass

            
        

    return render(request,'home.html')

def dashboard(request):
    worldcontext={'data':0}
    if request.method=="POST":
        selecteddata = request.POST['data']
        if selecteddata == 'worldwide':
            responsew = requests.get("https://api.covid19api.com/summary")
            dt = responsew.json()
            #global data
            Global_NewConfirmed= dt['Global']['NewConfirmed']
            Global_TotalConfirmed= dt['Global']['TotalConfirmed']
            Global_NewDeaths= dt['Global']['NewDeaths']
            Global_TotalDeaths= dt['Global']['TotalDeaths']
            Global_NewRecovered= dt['Global']['NewRecovered']
            Global_TotalRecovered= dt['Global']['TotalRecovered']
            Global_Date= dt['Global']['Date']

            for key,value in dt.items():
                country = dt['Countries']

            worldcontext = {'newconf':Global_NewConfirmed,'totconf':Global_TotalConfirmed,'newdeaths':Global_NewDeaths,
                            'totdeath':Global_TotalDeaths,'newrec':Global_NewRecovered, 
                            'totrec':Global_TotalRecovered, 'date':Global_Date, 'country':country}
            
        if selecteddata == 'india':
            responsei = requests.get('https://data.covid19india.org/v4/min/data.min.json')
            dti = responsei.json()

    # url = "https://api.covid19api.com/summary" 
    # response = urlopen(url) 
    # dt= json.loads(response.read()) 
    
    return render(request,'dashboard.html',worldcontext)

def main(request):
    return render(request,'main.html')

def prediction(request):
    cursor_ind = connection.cursor()
    cursor_ind.execute("select * from covid_covid_india_data")
    row_ind = cursor_ind.fetchall()

    data_ind=pd.DataFrame(row_ind,columns=['index_no','date','ordinal_date','total_confirmed','total_active','total_recovered','total_deaths','total_tested',
    'delta_confirmed','delta_active','delta_recovered','delta_deaths','delta_tested','delta7_confirmed','delta7_active','delta7_recovered',
    'delta7_deaths','delta7_tested','total_vaccinated1','total_vaccinated2','delta_vaccinated1','delta_vaccinated2','delta7_vaccinated1','delta7_vaccinated2',
    'total_other','delta_other','delta7_other'])

    train_data,test_data=train_test_split(data_ind,train_size=0.95,shuffle=False)

    train_features=train_data[['ordinal_date','total_tested']].to_numpy()
    train_labels=train_data['total_confirmed']

    test_features=test_data[['ordinal_date','total_tested']]
    test_labels=test_data['total_confirmed']


    model1=LinearRegression()
    model2=RandomForestRegressor(random_state=1)
    model3=DecisionTreeRegressor()
    model4=ExtraTreeRegressor()

    model1.fit(train_features,train_labels)
    model2.fit(train_features,train_labels)
    model3.fit(train_features,train_labels)
    model4.fit(train_features,train_labels)

    lin_train_predict=model1.predict(train_features)
    lin_test_predict=model1.predict(test_features)


    ran_train_predict=model2.predict(train_features)
    dtr_train_predict=model3.predict(train_features)
    etr_train_predict=model4.predict(train_features)


    MAE_lin_test = mean_absolute_error(test_labels,lin_test_predict)
    MSE_lin_test = numpy.sqrt(mean_squared_error(test_labels,lin_test_predict))
    
    MAE_ran_train = mean_absolute_error(train_labels,ran_train_predict)
    MSE_ran_train = numpy.sqrt(mean_squared_error(train_labels,ran_train_predict))
    
    MAE_lin_train = mean_absolute_error(train_labels,lin_train_predict)
    MSE_lin_train = numpy.sqrt(mean_squared_error(train_labels,lin_train_predict))

    MAE_dtr_train = mean_absolute_error(train_labels,dtr_train_predict)
    MSE_dtr_train = numpy.sqrt(mean_squared_error(train_labels,dtr_train_predict))

    MAE_etr_train = mean_absolute_error(train_labels,etr_train_predict)
    MSE_etr_train = numpy.sqrt(mean_squared_error(train_labels,etr_train_predict))


    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=train_data['date'],y=train_labels,mode = 'lines',name='train_labels1'))
    fig1.add_trace(go.Scatter(x=train_data['date'],y=lin_train_predict,mode = 'lines',name='train_data_predicted1'))
    fig1.update_layout(title = 'Comparing actual vs predicted total confirmed',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg1 = fig1.to_html(full_html=False)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=train_data['date'],y=train_labels,mode = 'lines',name='train_labels1'))
    fig3.add_trace(go.Scatter(x=train_data['date'],y=ran_train_predict,mode = 'lines',name='train_data_predicted1'))
    fig3.update_layout(title = 'Comparing actual vs predicted total confirmed',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg3 = fig3.to_html(full_html=False)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=train_data['date'],y=train_labels,mode = 'lines',name='train_labels1'))
    fig4.add_trace(go.Scatter(x=train_data['date'],y=dtr_train_predict,mode = 'lines',name='train_data_predicted1'))
    fig4.update_layout(title = 'Comparing actual vs predicted total confirmed',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg4 = fig4.to_html(full_html=False)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=train_data['date'],y=train_labels,mode = 'lines',name='train_labels1'))
    fig5.add_trace(go.Scatter(x=train_data['date'],y=etr_train_predict,mode = 'lines',name='train_data_predicted1'))
    fig5.update_layout(title = 'Comparing actual vs predicted total confirmed',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg5 = fig5.to_html(full_html=False)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=test_data['date'],y=test_labels,mode = 'lines',name='test_labels1'))
    fig2.add_trace(go.Scatter(x=test_data['date'],y=lin_test_predict,mode = 'lines',name='test_data_predicted1'))
    fig2.add_trace(go.Scatter(x=train_data['date'],y=train_labels,mode = 'lines',name='train_labels'))
    fig2.update_layout(title = 'Comparing actual vs predicted total confirmed',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg2 = fig2.to_html(full_html=False)


# Forcasting

    for_features = data_ind[['date','total_confirmed']]
    for_train_features = train_data[['date','total_confirmed']]
    for_test_features = test_data[['date','total_confirmed']]

    for_features['date'] = pd.to_datetime(for_features['date'])
    for_train_features['date'] = pd.to_datetime(for_train_features['date'])
    for_test_features['date'] = pd.to_datetime(for_test_features['date'])

    for_features = for_features.set_index('date')
    for_train_features = for_train_features.set_index('date')
    for_test_features = for_test_features.set_index('date')

    arima_mode = ARIMA(for_train_features,order=(5,1,0))
    arima_mode_all = ARIMA(for_features,order=(5,2,2))

    arima_mode_fit = arima_mode.fit()
    arima_mode_fit_all = arima_mode_all.fit()

    arima1_train_predicted = arima_mode_fit.predict()
    arima_mode_fit_all_pred = arima_mode_fit_all.predict()
    test_arima_pred = arima_mode_fit.forecast(steps=len(for_test_features))

    arima_forcast = arima_mode_fit_all.forecast(steps=7)

    MAE_ar = mean_absolute_error(arima1_train_predicted,train_labels)
    MSE_ar = numpy.sqrt(mean_squared_error(arima1_train_predicted,train_labels))

    MAE_ar_test = mean_absolute_error(test_arima_pred,test_labels)
    MSE_ar_test = numpy.sqrt(mean_squared_error(test_arima_pred,test_labels))

    sarima_model_all = SARIMAX(for_features,order=(5,1,2))
    sarima_model_train = SARIMAX(for_train_features,order=(5,1,2),seasonal_order=(1,1,1,12))

    sarima_model_all_fit = sarima_model_all.fit()
    sarima_model_train_fit = sarima_model_train.fit()

    test_sarima = sarima_model_train_fit.forecast(steps=len(for_test_features))

    sarima_train_pred = sarima_model_train_fit.predict()
    sarima_all = sarima_model_all_fit.forecast(steps=7)

    MAE_sr = mean_absolute_error(sarima_train_pred,train_labels)
    MSE_sr = numpy.sqrt(mean_squared_error(sarima_train_pred,train_labels))

    MAE_sr_test = mean_absolute_error(test_sarima,test_labels)
    MSE_sr_test = numpy.sqrt(mean_squared_error(test_sarima,test_labels))

    date=738044
    date_range=[]
    l=[]
    for i in range(1,8):
        l.append([date+i])
        date_range.append(datetime.date.fromordinal(date+i))

    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_confirmed'],mode = 'lines',name='total confirmed'))
    fig1.add_trace(go.Scatter(x=test_data['date'],y=test_sarima,mode = 'lines',name='test_sarima'))
    fig1.add_trace(go.Scatter(x=train_data['date'],y=sarima_train_pred,mode = 'lines',name='sarima_train_pred'))
    fig1.add_trace(go.Scatter(x=date_range,y=sarima_all,mode = 'lines',name='sarima_all'))
    fig1.update_layout(title = 'Comparing actual vs predicted total confirmed',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    f1 = fig1.to_html(full_html=False)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_confirmed'],mode = 'lines',name='total confirmed'))
    fig2.add_trace(go.Scatter(x=test_data['date'],y=test_arima_pred,mode = 'lines',name='test_arima_pred'))
    fig2.add_trace(go.Scatter(x=train_data['date'],y=arima1_train_predicted,mode = 'lines',name='arima1_train_predicted'))
    fig2.add_trace(go.Scatter(x=date_range,y=arima_forcast,mode = 'lines',name='arima_forcast'))
    fig2.update_layout(title = 'Comparing actual vs predicted total confirmed',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    f2 = fig2.to_html(full_html=False)

    context={'MAE_lin_test':MAE_lin_test,'MSE_lin_test':MSE_lin_test,'fg2':fg2,
    'MAE_lin_train':MAE_lin_train,'MSE_lin_train':MSE_lin_train,'fg1':fg1,
    'MAE_ran_train':MAE_ran_train,'MSE_ran_train':MSE_ran_train,'fg3':fg3,
    'MAE_dtr_train':MAE_dtr_train,'MSE_dtr_train':MSE_dtr_train,'fg4':fg4,
    'MAE_etr_train':MAE_etr_train,'MSE_etr_train':MSE_etr_train,'fg5':fg5,
    'MAE_ar':MAE_ar,'MSE_ar':MSE_ar,'f2':f2,
    'MAE_sr':MAE_sr,'MSE_sr':MSE_sr,'f1':f1,
    'MAE_sr_test':MAE_sr_test,'MSE_sr_test':MSE_sr_test,'MAE_ar_test':MAE_ar_test,'MSE_ar_test':MSE_ar_test
    }
    return render(request,'prediction.html',context)

def analysis_ind(request):
    context = {}
    # India data fetched
    cursor_ind = connection.cursor()
    cursor_ind.execute("select * from covid_covid_india_data")
    row_ind = cursor_ind.fetchall()

    data_ind=pd.DataFrame(row_ind,columns=['index_no','date','ordinal_date','total_confirmed','total_active','total_recovered','total_deaths','total_tested',
    'delta_confirmed','delta_active','delta_recovered','delta_deaths','delta_tested','delta7_confirmed','delta7_active','delta7_recovered',
    'delta7_deaths','delta7_tested','total_vaccinated1','total_vaccinated2','delta_vaccinated1','delta_vaccinated2','delta7_vaccinated1','delta7_vaccinated2',
    'total_other','delta_other','delta7_other'])




    # print(data)
    # graph - india
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_confirmed'],mode = 'lines',name='Total Confirmed'))
    fig1.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_recovered'],mode = 'lines',name='Total Recovered'))
    fig1.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_deaths'],mode = 'lines',name='Total Deaths'))
    fig1.update_layout(title = 'Covid19 Cases in India (Total Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg1 = fig1.to_html(full_html=False)

    # graph- death
    fig2= go.Figure()
    fig2.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_deaths'],mode = 'lines',name='Total Deaths'))
    fig2.update_layout(title = 'Covid19 - Deaths Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg2 = fig2.to_html(full_html=False)

    # graph- confirmed
    fig3= go.Figure()
    fig3.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_confirmed'],mode = 'lines',name='Total Confirmed'))
    fig3.update_layout(title = 'Covid19 - Confirmed Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg3 = fig3.to_html(full_html=False)

    # graph- recovered
    fig4= go.Figure()
    fig4.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_recovered'],mode = 'lines',name='Total Recovered'))
    fig4.update_layout(title = 'Covid19 - Recovered Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg4 = fig4.to_html(full_html=False)

    # graph- active
    fig5= go.Figure()
    fig5.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_active'],mode = 'lines',name='Active'))
    fig5.update_layout(title = 'Covid19 - Active Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg5 = fig5.to_html(full_html=False)

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_confirmed'],mode = 'lines',name='delta Confirmed'))
    fig6.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_recovered'],mode = 'lines',name='delta Recovered'))
    fig6.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_deaths'],mode = 'lines',name='delta Deaths'))
    fig6.update_layout(title = 'Covid19 Cases in India (Daywise data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg6 = fig6.to_html(full_html=False)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_confirmed'],mode = 'lines',name='delta Confirmed'))
    fig7.update_layout(title = 'Covid19 - Confirmed Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg7 = fig7.to_html(full_html=False)

    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_deaths'],mode = 'lines',name='delta Deaths'))
    fig8.update_layout(title = 'Covid19 - Deaths Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg8 = fig8.to_html(full_html=False)

    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_recovered'],mode = 'lines',name='delta Recovered'))
    fig9.update_layout(title = 'Covid19 - Recovered Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg9 = fig9.to_html(full_html=False)

    fig10 = go.Figure()
    fig10.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_active'],mode = 'lines',name='delta Active'))
    fig10.update_layout(title = 'Covid19 - Active Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
    fg10 = fig10.to_html(full_html=False)


    # graph- active
    fig_vc1= go.Figure()
    fig_vc1.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_vaccinated1'],mode = 'lines',name='vaccination1'))
    fig_vc1.update_layout(title = 'Covid19 - vaccination dose 1',xaxis_tickfont_size=15,yaxis=dict(title = 'total vaccination dose 1'),xaxis = dict(title = 'Date'))
    fg_v1 = fig_vc1.to_html(full_html=False)

    fig_vc2= go.Figure()
    fig_vc2.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_vaccinated2'],mode = 'lines',name='vaccination2'))
    fig_vc2.update_layout(title = 'Covid19 - vaccination dose 2',xaxis_tickfont_size=15,yaxis=dict(title = 'total vaccination dose 2'),xaxis = dict(title = 'Date'))
    fg_v2 = fig_vc2.to_html(full_html=False)

    fig_vc3= go.Figure()
    fig_vc3.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_vaccinated2'],mode = 'lines',name='vaccination1'))
    fig_vc3.update_layout(title = 'Covid19 - vaccination dose 1',xaxis_tickfont_size=15,yaxis=dict(title = 'daywise vaccination dose 2'),xaxis = dict(title = 'Date'))
    fg_v3 = fig_vc3.to_html(full_html=False)

    fig_vc4= go.Figure()
    fig_vc4.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_vaccinated2'],mode = 'lines',name='vaccination2'))
    fig_vc4.update_layout(title = 'Covid19 - vaccination dose 2',xaxis_tickfont_size=15,yaxis=dict(title = 'daywise vaccination dose 2'),xaxis = dict(title = 'Date'))
    fg_v4 = fig_vc4.to_html(full_html=False)

    
    fig_tt1= go.Figure()
    fig_tt1.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['total_tested'],mode = 'lines',name='tested'))
    fig_tt1.update_layout(title = 'Covid19 - total tested',xaxis_tickfont_size=15,yaxis=dict(title = 'total tested'),xaxis = dict(title = 'Date'))
    fg_tt1 = fig_tt1.to_html(full_html=False)

    fig_dt1= go.Figure()
    fig_dt1.add_trace(go.Scatter(x=data_ind['date'],y=data_ind['delta_tested'],mode = 'lines',name='tested'))
    fig_dt1.update_layout(title = 'Covid19 - delta tested',xaxis_tickfont_size=15,yaxis=dict(title = 'delta tested'),xaxis = dict(title = 'Date'))
    fg_dt1 = fig_dt1.to_html(full_html=False)


    context = {'fig1':fg1,'fig2':fg2, 'fig3':fg3, 'fig4':fg4, 'fig5':fg5, 'fig6':fg6, 
    'fig7':fg7, 'fig8':fg8, 'fig9':fg9, 'fig10':fg10,'fg_v1':fg_v1 ,'fg_v2':fg_v2, 'fg_v3':fg_v3,'fg_v4':fg_v4,'fg_tt1':fg_tt1,'fg_dt1':fg_dt1}
    
    return render(request,'analysis_ind.html',context)

def analysis_st(request):
    context={}
    # states data fetched
    stcode=''
    if request.method == 'GET':
        
        cursor_st = connection.cursor()
        cursor_st.execute("select * from covid_covid_state_data")
        row_st = cursor_st.fetchall()

        data_st=pd.DataFrame(row_st,columns=['index_no','date','ordinal_date','state_name','total_confirmed','total_active','total_recovered','total_deaths','total_tested',
        'delta_confirmed','delta_active','delta_recovered','delta_deaths','delta_tested','delta7_confirmed','delta7_active','delta7_recovered',
        'delta7_deaths','delta7_tested','total_vaccinated1','total_vaccinated2','delta_vaccinated1','delta_vaccinated2','delta7_vaccinated1','delta7_vaccinated2',
        'total_other','delta_other','delta7_other'])
        print(data_st.head())
        stcode = request.GET.get('st_cd')
        # print(stcode)
        state_data = data_st[(data_st['state_name']==stcode)]
        # print(state_data)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_confirmed'],mode = 'lines',name='Total Confirmed'))
        fig1.update_layout(title = 'Covid19 - Total Confirmed Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg1 = fig1.to_html(full_html=False)


        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_recovered'],mode = 'lines',name='Total Recovered'))
        fig2.update_layout(title = 'Covid19 - Total Recovered Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg2 = fig2.to_html(full_html=False)


        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_deaths'],mode = 'lines',name='Total Deaths'))
        fig3.update_layout(title = 'Covid19 - Total Deaths Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg3 = fig3.to_html(full_html=False)


        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_active'],mode = 'lines',name='Total Active'))
        fig4.update_layout(title = 'Covid19 - Total Active Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg4 = fig4.to_html(full_html=False)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_confirmed'],mode = 'lines',name='delta Confirmed'))
        fig5.update_layout(title = 'Covid19 - Confirmed Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg5 = fig5.to_html(full_html=False)
        
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_recovered'],mode = 'lines',name='delta Recovered'))
        fig6.update_layout(title = 'Covid19 - Recovered Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg6 = fig6.to_html(full_html=False)
        
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_deaths'],mode = 'lines',name='delta Deaths'))
        fig7.update_layout(title = 'Covid19 - Deaths Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg7 = fig7.to_html(full_html=False)
        
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_active'],mode = 'lines',name='delta Active'))
        fig8.update_layout(title = 'Covid19 - Active Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg8 = fig8.to_html(full_html=False)

        fig_vc1= go.Figure()
        fig_vc1.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_vaccinated1'],mode = 'lines',name='vaccination1'))
        fig_vc1.update_layout(title = 'Covid19 - vaccination dose 1',xaxis_tickfont_size=15,yaxis=dict(title = 'total vaccination dose 1'),xaxis = dict(title = 'Date'))
        fg_v1 = fig_vc1.to_html(full_html=False)

        fig_vc2= go.Figure()
        fig_vc2.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_vaccinated2'],mode = 'lines',name='vaccination1'))
        fig_vc2.update_layout(title = 'Covid19 - vaccination dose 2',xaxis_tickfont_size=15,yaxis=dict(title = 'total vaccination dose 2'),xaxis = dict(title = 'Date'))
        fg_v2 = fig_vc2.to_html(full_html=False)

        fig_vc3= go.Figure()
        fig_vc3.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_vaccinated1'],mode = 'lines',name='vaccination1'))
        fig_vc3.update_layout(title = 'Covid19 - vaccination dose 1',xaxis_tickfont_size=15,yaxis=dict(title = 'delta vaccination dose 1'),xaxis = dict(title = 'Date'))
        fg_v3 = fig_vc3.to_html(full_html=False)

        fig_vc4= go.Figure()
        fig_vc4.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_vaccinated2'],mode = 'lines',name='vaccination1'))
        fig_vc4.update_layout(title = 'Covid19 - vaccination dose 2',xaxis_tickfont_size=15,yaxis=dict(title = 'delta vaccination dose 2'),xaxis = dict(title = 'Date'))
        fg_v4 = fig_vc4.to_html(full_html=False)

        fig_tt1= go.Figure()
        fig_tt1.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_tested'],mode = 'lines',name='tested'))
        fig_tt1.update_layout(title = 'Covid19 - total tested',xaxis_tickfont_size=15,yaxis=dict(title = 'total tested'),xaxis = dict(title = 'Date'))
        fg_tt1 = fig_tt1.to_html(full_html=False)

        fig_dt1= go.Figure()
        fig_dt1.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_tested'],mode = 'lines',name='tested'))
        fig_dt1.update_layout(title = 'Covid19 - delta tested',xaxis_tickfont_size=15,yaxis=dict(title = 'delta tested'),xaxis = dict(title = 'Date'))
        fg_dt1 = fig_dt1.to_html(full_html=False)

        temp = data_st.groupby(['state_name','date'])['total_confirmed','total_recovered','total_deaths','total_active'].sum().reset_index()
        state = temp['state_name'].unique()
        stloc = pd.read_csv('./covid/static/files/state_loc.csv')
        stloc.reset_index(inplace=True)
        stloc.drop(['Unnamed: 0'],axis=1,inplace=True)
        deta = pd.merge(temp,stloc,how='inner',left_on='state_name',right_on='State.Name')
        deta.drop(['index','State.Name'],inplace=True,axis=1)
        deta['date'] = deta['date'].astype(str)

        fig_map = px.density_mapbox(deta,lat='latitude',lon='longitude',hover_name='state_name',hover_data=['total_confirmed','total_recovered','total_deaths','total_active'],animation_frame='date',color_continuous_scale='Portland',radius=7,zoom=0,height=700)
        fig_map.update_layout(title='Covid Statewise')
        fig_map.update_layout(mapbox_style='open-street-map',mapbox_center_lon=0)
        fig_mp = fig_map.to_html(full_html=False)

        deta = deta[deta['date']==max(deta['date'])]

        fig_c = px.bar(deta.sort_values('total_confirmed'),x='total_confirmed',y='state_name',
                    text='total_confirmed',orientation='h',color_discrete_sequence=[dth])

        fig_a = px.bar(deta.sort_values('total_recovered'),x='total_recovered',y='state_name',
                    text='total_recovered',orientation='h',color_discrete_sequence=[act])

        fig_b = px.bar(deta.sort_values('total_active'),x='total_active',y='state_name',
                    text='total_active',orientation='h',color_discrete_sequence=[dth])

        fig_d = px.bar(deta.sort_values('total_deaths'),x='total_deaths',y='state_name',
                    text='total_deaths',orientation='h',color_discrete_sequence=[act])

        fig_abcd = make_subplots(rows=2,cols=2,shared_xaxes=False,horizontal_spacing=0.14,
                        vertical_spacing=0.1,
                        subplot_titles=('confirmed','recovered','active','deaths'))

        fig_abcd.add_trace(fig_c['data'][0],row=1,col=1)
        fig_abcd.add_trace(fig_a['data'][0],row=1,col=2)
        fig_abcd.add_trace(fig_b['data'][0],row=2,col=1)
        fig_abcd.add_trace(fig_d['data'][0],row=2,col=2)
        fig_abcd.update_layout(height=2000)
        fig_ad = fig_abcd.to_html(full_html=False)

        context={'st_cd':stcode ,'fig1':fg1,'fig2':fg2, 'fig3':fg3,'fig4':fg4,'fig5':fg5,'fig6':fg6,
        'fig7':fg7,'fig8':fg8,
        'fg_v1':fg_v1 ,'fg_v2':fg_v2, 'fg_v3':fg_v3,'fg_v4':fg_v4,
        'fg_tt1':fg_tt1,'fg_dt1':fg_dt1, 'fig_ad':fig_ad,'fig_mp':fig_mp}
    return render(request,'analysis_st.html',context)

def analysis_dt(request):
    context={}
    dtcode=''
    if request.method == 'GET':
        # districts data fetched
        cursor_dt = connection.cursor()
        cursor_dt.execute("select * from covid_covid_district_data")
        row_dt = cursor_dt.fetchall()

        data_dt=pd.DataFrame(row_dt,columns=['index_no','date','ordinal_date','state_name','district_name','total_confirmed','total_active','total_recovered','total_deaths',
        'delta_confirmed','delta_active','delta_recovered','delta_deaths','delta7_confirmed','delta7_active','delta7_recovered',
        'delta7_deaths','total_vaccinated1','total_vaccinated2','delta_vaccinated1','delta_vaccinated2','delta7_vaccinated1','delta7_vaccinated2',
        'total_other','delta_other','delta7_other'])


        dtcode = request.GET.get('dt_cd')
        # print(stcode)
        if dtcode != None:
            dtcode = dtcode.lower()
        state_data = data_dt[(data_dt['district_name'].str.lower()==dtcode)]
        # print(state_data)
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_confirmed'],mode = 'lines',name='Total Confirmed'))
        fig1.update_layout(title = 'Covid19 - Total Confirmed Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg1 = fig1.to_html(full_html=False)


        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_recovered'],mode = 'lines',name='Total Recovered'))
        fig2.update_layout(title = 'Covid19 - Total Recovered Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg2 = fig2.to_html(full_html=False)


        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_deaths'],mode = 'lines',name='Total Deaths'))
        fig3.update_layout(title = 'Covid19 - Total Deaths Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg3 = fig3.to_html(full_html=False)


        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_active'],mode = 'lines',name='Total Active'))
        fig4.update_layout(title = 'Covid19 - Total Active Cases',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg4 = fig4.to_html(full_html=False)
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_confirmed'],mode = 'lines',name='delta Confirmed'))
        fig5.update_layout(title = 'Covid19 - Confirmed Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg5 = fig5.to_html(full_html=False)
        
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_recovered'],mode = 'lines',name='delta Recovered'))
        fig6.update_layout(title = 'Covid19 - Recovered Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg6 = fig6.to_html(full_html=False)
        
        fig7 = go.Figure()
        fig7.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_deaths'],mode = 'lines',name='delta Deaths'))
        fig7.update_layout(title = 'Covid19 - Deaths Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg7 = fig7.to_html(full_html=False)
        
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_active'],mode = 'lines',name='delta Active'))
        fig8.update_layout(title = 'Covid19 - Active Cases (Daywise Data)',xaxis_tickfont_size=15,yaxis=dict(title = 'Number of cases'),xaxis = dict(title = 'Date'))
        fg8 = fig8.to_html(full_html=False)


        fig_vc1= go.Figure()
        fig_vc1.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_vaccinated1'],mode = 'lines',name='vaccination1'))
        fig_vc1.update_layout(title = 'Covid19 - vaccination dose 1',xaxis_tickfont_size=15,yaxis=dict(title = 'total vaccination dose 1'),xaxis = dict(title = 'Date'))
        fg_v1 = fig_vc1.to_html(full_html=False)

        fig_vc2= go.Figure()
        fig_vc2.add_trace(go.Scatter(x=state_data['date'],y=state_data['total_vaccinated2'],mode = 'lines',name='vaccination1'))
        fig_vc2.update_layout(title = 'Covid19 - vaccination dose 2',xaxis_tickfont_size=15,yaxis=dict(title = 'total vaccination dose 2'),xaxis = dict(title = 'Date'))
        fg_v2 = fig_vc2.to_html(full_html=False)

        fig_vc3= go.Figure()
        fig_vc3.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_vaccinated1'],mode = 'lines',name='vaccination1'))
        fig_vc3.update_layout(title = 'Covid19 - vaccination dose 1',xaxis_tickfont_size=15,yaxis=dict(title = 'delta vaccination dose 1'),xaxis = dict(title = 'Date'))
        fg_v3 = fig_vc3.to_html(full_html=False)

        fig_vc4= go.Figure()
        fig_vc4.add_trace(go.Scatter(x=state_data['date'],y=state_data['delta_vaccinated2'],mode = 'lines',name='vaccination1'))
        fig_vc4.update_layout(title = 'Covid19 - vaccination dose 2',xaxis_tickfont_size=15,yaxis=dict(title = 'delta vaccination dose 2'),xaxis = dict(title = 'Date'))
        fg_v4 = fig_vc4.to_html(full_html=False)

        context={'st_cd':dtcode ,'fig1':fg1,'fig2':fg2, 'fig3':fg3,'fig4':fg4,'fig5':fg5,'fig6':fg6,'fig7':fg7,'fig8':fg8,
        'fg_v1':fg_v1 ,'fg_v2':fg_v2, 'fg_v3':fg_v3,'fg_v4':fg_v4}
    return render(request,'analysis_dt.html',context)