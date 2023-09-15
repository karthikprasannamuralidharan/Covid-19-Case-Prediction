import pandas 
import numpy
import datetime
import mysql.connector
import json
from covid.models import *
from django.test import TestCase
 
def processing_for_india_table(key,value):
    year=key[0:4]
    month=key[5:7]
    day=key[8:10]
    date=datetime.date(int(year),int(month),int(day))
    ordinaldate=datetime.datetime.toordinal(date)
    # print(date)
    # print(ordinaldate)
    # print(key)
    total_confirmed=0
    total_recovered=0
    total_deaths=0
    total_tested=0
    total_active=0
    total_other=0
    delta_confirmed=0
    delta_recovered=0
    delta_deaths=0
    delta_tested=0
    delta_active=0
    delta_other=0
    delta7_confirmed=0
    delta7_recovered=0
    delta7_deaths=0
    delta7_tested=0
    delta7_active=0
    delta7_other=0
    total_vaccinated1=0
    total_vaccinated2=0
    delta_vaccinated1=0
    delta_vaccinated2=0
    delta7_vaccinated1=0
    delta7_vaccinated2=0
    # print(value.keys())
    try:
        # print(value['TT'])
        if value['TT']!=None:
            try:
                # print(value['TT']['delta'])
                if value['TT']['delta']!=None:
                    try:
                        delta_confirmed=value['TT']['delta']['confirmed']
                        # print(value['TT']['delta']['confirmed'])
                    except:
                        pass
                    try:
                        delta_recovered=value['TT']['delta']['recovered']
                        # print(value['TT']['delta']['recovered'])
                    except:
                        pass
                    try:
                        delta_deaths=value['TT']['delta']['deceased']
                        # print(value['TT']['delta']['deceased'])
                    except:
                        pass
                    try:
                        delta_tested=value['TT']['delta']['tested']
                        # print(value['TT']['delta']['tested'])
                    except:
                        pass
                    try:
                        delta_other=value['TT']['delta']['other']
                        # print(value['TT']['delta']['other'])
                    except:
                        pass
                    try:
                        delta_vaccinated1=value['TT']['delta']['vaccinated1']
                        # print(value['TT']['delta']['vaccinated1'])
                    except:
                        pass
                    try:
                        delta_vaccinated2=value['TT']['delta']['vaccinated2']
                        # print(value['TT']['delta']['vaccinated2'])
                    except:
                        pass
                    delta_active=delta_confirmed-delta_recovered-delta_deaths-delta_other
            except:
                pass
            try:
                # print(value['TT']['delta7'])
                if value['TT']['delta7']!=None:
                    try:
                        delta7_confirmed=value['TT']['delta7']['confirmed']
                        # print(value['TT']['delta7']['confirmed'])
                    except:
                        pass
                    try:
                        delta7_recovered=value['TT']['delta7']['recovered']
                        # print(value['TT']['delta7']['recovered'])
                    except:
                        pass
                    try:
                        delta7_deaths=value['TT']['delta7']['deceased']
                        # print(value['TT']['delta7']['deceased'])
                    except:
                        pass
                    try:
                        delta7_tested=value['TT']['delta7']['tested']
                        # print(value['TT']['delta7']['tested'])
                    except:
                        pass
                    try:
                        delta7_other=value['TT']['delta7']['other']
                        # print(value['TT']['delta7']['other'])
                    except:
                        pass
                    try:
                        delta7_vaccinated1=value['TT']['delta7']['vaccinated1']
                        # print(value['TT']['delta7']['vaccinated1'])
                    except:
                        pass
                    try:
                        delta7_vaccinated2=value['TT']['delta7']['vaccinated2']
                        # print(value['TT']['delta7']['vaccinated2'])
                    except:
                        pass
                    delta7_active=delta7_confirmed-delta7_recovered-delta7_deaths-delta7_other
            except:
                pass
            try:
                # print(value['TT']['total'])
                if value['TT']['total']!=None:
                    try:
                        total_confirmed=value['TT']['total']['confirmed']
                        # print(value['TT']['total']['confirmed'])
                    except:
                        pass
                    try:
                        total_recovered=value['TT']['total']['recovered']
                        # print(value['TT']['total']['recovered'])
                    except:
                        pass
                    try:
                        total_deaths=value['TT']['total']['deceased']
                        # print(value['TT']['total']['deceased'])
                    except:
                        pass
                    try:
                        total_tested=value['TT']['total']['tested']
                        # print(value['TT']['total']['tested'])
                    except:
                        pass
                    try:
                        total_other=value['TT']['total']['other']
                        # print(value['TT']['total']['other'])
                    except:
                        pass
                    try:
                        total_vaccinated1=value['TT']['total']['vaccinated1']
                        # print(value['TT']['total']['vaccinated1'])
                    except:
                        pass
                    try:
                        total_vaccinated2=value['TT']['total']['vaccinated2']
                        # print(value['TT']['total']['vaccinated2'])
                    except:
                        pass
                    total_active=total_confirmed-total_recovered-total_deaths-total_other
            except:
                pass

    except:
        pass
    dt = covid_india_data(date=date,ordinal_date=ordinaldate,total_confirmed=total_confirmed,total_active=total_active,total_recovered=total_recovered,total_deaths=total_deaths,total_tested=total_tested,delta_confirmed=delta_confirmed,delta_active=delta_active,delta_recovered=delta_recovered,delta_deaths=delta_deaths,delta_tested=delta_tested,delta7_confirmed=delta7_confirmed,delta7_active=delta7_active,delta7_recovered=delta7_recovered,delta7_deaths=delta7_deaths,delta7_tested=delta7_tested,total_vaccinated1=total_vaccinated1,total_vaccinated2=total_vaccinated2,delta_vaccinated1=delta_vaccinated1,delta_vaccinated2=delta_vaccinated2,delta7_vaccinated1=delta7_vaccinated1,delta7_vaccinated2=delta7_vaccinated2,total_other=total_other,delta_other=delta_other,delta7_other=delta7_other)
    dt.save()
    print("done : ",date)
     


    data=[]
    with open('./covid/static/data/jsondata.json') as f:
        data=json.load(fp=f)

    for key,value in data.items():
        processing_for_india_table(key,value)


    with open('./covid/static/data/jsondata2.json') as f:
        data=json.load(fp=f)

    for i in data:
        # print(i)
        for key,value in i.items():
            processing_for_india_table(key,value)
        
    # sql = india_data.objects.all()
    # print(sql)
    # with open('./cnf.env') as f:
    #     credentials=f.read()
    # credentials=credentials.split(" ")
    # con = mysql.connector.connect(username=credentials[0],password=credentials[1],host=credentials[2],port=credentials[3],database=credentials[4])
    # query=con.cursor()
    # sql="SELECT * FROM covid_india_data"
    # query.execute(sql)
    # result=query.fetchall()
    # column=[columns[0] for columns in query.description]

    # data=pd.DataFrame(result,columns=column)
    # print(data)
# cursor = connection.cursor()

# cursor.execute("select * from covid_india_data")
# row = cursor.fetchall()

# data=pd.DataFrame(row,columns=['index_no','date','ordinal_date','total_confirmed','total_active','total_recovered','total_deaths','total_tested',
# 'delta_confirmed','delta_active','delta_recovered','delta_deaths','delta_tested','delta7_confirmed','delta7_active','delta7_recovered',
# 'delta7_deaths','delta7_tested','total_vaccinated1','total_vaccinated2','delta_vaccinated1','delta_vaccinated2','delta7_vaccinated1','delta7_vaccinated2',
# 'total_other','delta_other','delta7_other'])
# # print(data)
# # graph - india


def processing_for_state_table(key,value):
    year=key[0:4]
    month=key[5:7]
    day=key[8:10]
    date=datetime.date(int(year),int(month),int(day))
    ordinaldate=datetime.datetime.toordinal(date)
    # print(date)
    # print(ordinaldate)
    # print(key)
    # print(value)
    for index,element in value.items():
        # print(index)
        if index!="TT":
            # print(element)
            state=index
            total_confirmed=0
            total_recovered=0
            total_deaths=0
            total_tested=0
            total_active=0
            total_other=0
            delta_confirmed=0
            delta_recovered=0
            delta_deaths=0
            delta_tested=0
            delta_active=0
            delta_other=0
            delta7_confirmed=0
            delta7_recovered=0
            delta7_deaths=0
            delta7_tested=0
            delta7_active=0
            delta7_other=0
            total_vaccinated1=0
            total_vaccinated2=0
            delta_vaccinated1=0
            delta_vaccinated2=0
            delta7_vaccinated1=0
            delta7_vaccinated2=0
            try:
                # print(element['delta'])
                if element['delta']!=None:
                    try:
                        delta_confirmed=element['delta']['confirmed']
                        # print(element['delta']['confirmed'])
                    except:
                        pass
                    try:
                        delta_recovered=element['delta']['recovered']
                        # print(element['delta']['recovered'])
                    except:
                        pass
                    try:
                        delta_deaths=element['delta']['deceased']
                        # print(element['delta']['deceased'])
                    except:
                        pass
                    try:
                        delta_tested=element['delta']['tested']
                        # print(element['delta']['tested'])
                    except:
                        pass
                    try:
                        delta_other=element['delta']['other']
                        # print(element['delta']['other'])
                    except:
                        pass
                    try:
                        delta_vaccinated1=element['delta']['vaccinated1']
                        # print(element['delta']['vaccinated1'])
                    except:
                        pass
                    try:
                        delta_vaccinated2=element['delta']['vaccinated2']
                        # print(element['delta']['vaccinated2'])
                    except:
                        pass
                    delta_active=delta_confirmed-delta_recovered-delta_deaths-delta_other
            except:
                pass
            try:
                # print(element['delta7'])
                if element['delta7']!=None:
                    try:
                        delta7_confirmed=element['delta7']['confirmed']
                        # print(element['delta7']['confirmed'])
                    except:
                        pass
                    try:
                        delta7_recovered=element['delta7']['recovered']
                        # print(element['delta7']['recovered'])
                    except:
                        pass
                    try:
                        delta7_deaths=element['delta7']['deceased']
                        # print(element['delta7']['deceased'])
                    except:
                        pass
                    try:
                        delta7_tested=element['delta7']['tested']
                        # print(element['delta7']['tested'])
                    except:
                        pass
                    try:
                        delta7_other=element['delta7']['other']
                        # print(element['delta7']['other'])
                    except:
                        pass
                    try:
                        delta7_vaccinated1=element['delta7']['vaccinated1']
                        # print(element['delta7']['vaccinated1'])
                    except:
                        pass
                    try:
                        delta7_vaccinated2=element['delta7']['vaccinated2']
                        # print(element['delta7']['vaccinated2'])
                    except:
                        pass
                    delta7_active=delta7_confirmed-delta7_recovered-delta7_deaths-delta7_other
            except:
                pass
            try:
                # print(element['total'])
                if element['total']!=None:
                    try:
                        total_confirmed=element['total']['confirmed']
                        # print(element['total']['confirmed'])
                    except:
                        pass
                    try:
                        total_recovered=element['total']['recovered']
                        # print(element['total']['recovered'])
                    except:
                        pass
                    try:
                        total_deaths=element['total']['deceased']
                        # print(element['total']['deceased'])
                    except:
                        pass
                    try:
                        total_tested=element['total']['tested']
                        # print(element['total']['tested'])
                    except:
                        pass
                    try:
                        total_other=element['total']['other']
                        # print(element['total']['other'])
                    except:
                        pass
                    try:
                        total_vaccinated1=element['total']['vaccinated1']
                        # print(element['total']['vaccinated1'])
                    except:
                        pass
                    try:
                        total_vaccinated2=element['total']['vaccinated2']
                        # print(element['total']['vaccinated2'])
                    except:
                        pass
                    total_active=total_confirmed-total_recovered-total_deaths-total_other
            except:
                pass
            # print(state,total_confirmed,total_recovered,total_deaths,total_tested,total_active,total_other,delta_confirmed,delta_recovered,delta_deaths,delta_tested,delta_active,delta_other,delta7_confirmed,delta7_recovered,delta7_deaths,delta7_tested,delta7_active,delta7_other,total_vaccinated1,total_vaccinated2,delta_vaccinated1,delta_vaccinated2,delta7_vaccinated1,delta7_vaccinated2, sep="   ")
            dt = covid_india_data(date=date,ordinal_date=ordinaldate,total_confirmed=total_confirmed,total_active=total_active,total_recovered=total_recovered,total_deaths=total_deaths,total_tested=total_tested,delta_confirmed=delta_confirmed,delta_active=delta_active,delta_recovered=delta_recovered,delta_deaths=delta_deaths,delta_tested=delta_tested,delta7_confirmed=delta7_confirmed,delta7_active=delta7_active,delta7_recovered=delta7_recovered,delta7_deaths=delta7_deaths,delta7_tested=delta7_tested,total_vaccinated1=total_vaccinated1,total_vaccinated2=total_vaccinated2,delta_vaccinated1=delta_vaccinated1,delta_vaccinated2=delta_vaccinated2,delta7_vaccinated1=delta7_vaccinated1,delta7_vaccinated2=delta7_vaccinated2,total_other=total_other,delta_other=delta_other,delta7_other=delta7_other)
            dt.save()
            print("done : ",date)

data=[]
with open('./data/jsondata.json') as f:
    data=json.load(fp=f)

for key,value in data.items():
    processing_for_state_table(key,value)


data=[]
with open('./data/jsondata2.json') as f:
    data=json.load(fp=f)

for i in data:
    for key,value in i.items():
        processing_for_state_table(key,value)

