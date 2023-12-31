import pandas 
import numpy
import datetime
import mysql.connector
import json
from .models import *


def processing_for_district_table(key,value):
    year=key[0:4]
    month=key[5:7]
    day=key[8:10]
    # print(year,month,day)
    date=datetime.date(int(year),int(month),int(day))
    # print(date)
    ordinaldate=datetime.datetime.toordinal(date)
    # print(ordinaldate)
    for index,element in value.items():
        state=index
        # print(index)
        if index!="TT":
            try:
                if element['districts']!=None:
                    for objectkey,objects in element['districts'].items():
                        district=objectkey
                        total_confirmed=0
                        total_recovered=0
                        total_deaths=0
                        total_active=0
                        total_other=0
                        delta_confirmed=0
                        delta_recovered=0
                        delta_deaths=0
                        delta_active=0
                        delta_other=0
                        delta7_confirmed=0
                        delta7_recovered=0
                        delta7_deaths=0
                        delta7_active=0
                        delta7_other=0
                        total_vaccinated1=0
                        total_vaccinated2=0
                        delta_vaccinated1=0
                        delta_vaccinated2=0
                        delta7_vaccinated1=0
                        delta7_vaccinated2=0
                        try:
                            if objects['delta']!=None:
                                # print(objects['delta'])
                                try:
                                    delta_confirmed=objects['delta']['confirmed']
                                    # print(objects['delta']['confirmed'])
                                except:
                                    pass
                                try:
                                    delta_recovered=objects['delta']['recovered']
                                    # print(objects['delta']['recovered'])
                                except:
                                    pass
                                try:
                                    delta_deaths=objects['delta']['deceased']
                                    # print(objects['delta']['deceased'])
                                except:
                                    pass
                                try:
                                    delta_tested=objects['delta']['tested']
                                    # print(objects['delta']['tested'])
                                except:
                                    pass
                                try:
                                    delta_other=objects['delta']['other']
                                    # print(objects['delta']['other'])
                                except:
                                    pass
                                try:
                                    delta_vaccinated1=objects['delta']['vaccinated1']
                                    # print(objects['delta']['vaccinated1'])
                                except:
                                    pass
                                try:
                                    delta_vaccinated2=objects['delta']['vaccinated2']
                                    # print(objects['delta']['vaccinated2'])
                                except:
                                    pass
                                delta_active=delta_confirmed-delta_recovered-delta_deaths-delta_other
                        except:
                            pass
                        try:
                            # print(objects['delta7'])
                            if objects['delta7']!=None:
                                try:
                                    delta7_confirmed=objects['delta7']['confirmed']
                                    # print(objects['delta7']['confirmed'])
                                except:
                                    pass
                                try:
                                    delta7_recovered=objects['delta7']['recovered']
                                    # print(objects['delta7']['recovered'])
                                except:
                                    pass
                                try:
                                    delta7_deaths=objects['delta7']['deceased']
                                    # print(objects['delta7']['deceased'])
                                except:
                                    pass
                                try:
                                    delta7_tested=objects['delta7']['tested']
                                    # print(objects['delta7']['tested'])
                                except:
                                    pass
                                try:
                                    delta7_other=objects['delta7']['other']
                                    # print(objects['delta7']['other'])
                                except:
                                    pass
                                try:
                                    delta7_vaccinated1=objects['delta7']['vaccinated1']
                                    # print(objects['delta7']['vaccinated1'])
                                except:
                                    pass
                                try:
                                    delta7_vaccinated2=objects['delta7']['vaccinated2']
                                    # print(objects['delta7']['vaccinated2'])
                                except:
                                    pass
                                delta7_active=delta7_confirmed-delta7_recovered-delta7_deaths-delta7_other
                        except:
                            pass
                        try:
                            # print(objects['total'])
                            if objects['total']!=None:
                                try:
                                    total_confirmed=objects['total']['confirmed']
                                    # print(objects['total']['confirmed'])
                                except:
                                    pass
                                try:
                                    total_recovered=objects['total']['recovered']
                                    # print(objects['total']['recovered'])
                                except:
                                    pass
                                try:
                                    total_deaths=objects['total']['deceased']
                                    # print(objects['total']['deceased'])
                                except:
                                    pass
                                try:
                                    total_tested=objects['total']['tested']
                                    # print(objects['total']['tested'])
                                except:
                                    pass
                                try:
                                    total_other=objects['total']['other']
                                    # print(objects['total']['other'])
                                except:
                                    pass
                                try:
                                    total_vaccinated1=objects['total']['vaccinated1']
                                    # print(objects['total']['vaccinated1'])
                                except:
                                    pass
                                try:
                                    total_vaccinated2=objects['total']['vaccinated2']
                                    # print(objects['total']['vaccinated2'])
                                except:
                                    pass
                                total_active=total_confirmed-total_recovered-total_deaths-total_other
                        except:
                            pass
                        # print(state,district,total_confirmed,total_recovered,total_deaths,total_active,total_other,delta_confirmed,delta_recovered,delta_deaths,delta_active,delta_other,delta7_confirmed,delta7_recovered,delta7_deaths,delta7_active,delta7_other,total_vaccinated1,total_vaccinated2,delta_vaccinated1,delta_vaccinated2,delta7_vaccinated1,delta7_vaccinated2, sep="   ")
                        dt = covid_district_data(date=date,ordinal_date=ordinaldate,state_name=state,district_name=district,total_confirmed=total_confirmed,total_active=total_active,total_recovered=total_recovered,total_deaths=total_deaths,delta_confirmed=delta_confirmed,delta_active=delta_active,delta_recovered=delta_recovered,delta_deaths=delta_deaths,delta7_confirmed=delta7_confirmed,delta7_active=delta7_active,delta7_recovered=delta7_recovered,delta7_deaths=delta7_deaths,total_vaccinated1=total_vaccinated1,total_vaccinated2=total_vaccinated2,delta_vaccinated1=delta_vaccinated1,delta_vaccinated2=delta_vaccinated2,delta7_vaccinated1=delta7_vaccinated1,delta7_vaccinated2=delta7_vaccinated2,total_other=total_other,delta_other=delta_other,delta7_other=delta7_other)
                        dt.save()
                        
            except:
               pass
    print("done : ",date)


data=[]
with open('.covid/static/data/jsondata.json') as f:
    data=json.load(fp=f)

for key,value in data.items():
    processing_for_district_table(key,value)


data=[]
with open('.covid/static/data/jsondata2.json') as f:
    data=json.load(fp=f)

for i in data:
    # print(i)
    for key,value in i.items():
        processing_for_district_table(key,value)
