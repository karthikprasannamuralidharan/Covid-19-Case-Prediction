# Generated by Django 3.1.6 on 2021-10-01 14:06

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='covid_district_data',
            fields=[
                ('index_no', models.AutoField(primary_key=True, serialize=False)),
                ('date', models.DateField()),
                ('ordinal_date', models.IntegerField()),
                ('state_name', models.CharField(max_length=255)),
                ('district_name', models.CharField(max_length=255)),
                ('total_confirmed', models.IntegerField()),
                ('total_active', models.IntegerField()),
                ('total_recovered', models.IntegerField()),
                ('total_deaths', models.IntegerField()),
                ('total_tested', models.IntegerField()),
                ('delta_confirmed', models.IntegerField()),
                ('delta_active', models.IntegerField()),
                ('delta_recovered', models.IntegerField()),
                ('delta_deaths', models.IntegerField()),
                ('delta_tested', models.IntegerField()),
                ('delta7_confirmed', models.IntegerField()),
                ('delta7_active', models.IntegerField()),
                ('delta7_recovered', models.IntegerField()),
                ('delta7_deaths', models.IntegerField()),
                ('delta7_tested', models.IntegerField()),
                ('total_vaccinated1', models.IntegerField()),
                ('total_vaccinated2', models.IntegerField()),
                ('delta_vaccinated1', models.IntegerField()),
                ('delta_vaccinated2', models.IntegerField()),
                ('delta7_vaccinated1', models.IntegerField()),
                ('delta7_vaccinated2', models.IntegerField()),
                ('total_other', models.IntegerField()),
                ('delta_other', models.IntegerField()),
                ('delta7_other', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='covid_india_data',
            fields=[
                ('index_no', models.AutoField(primary_key=True, serialize=False)),
                ('date', models.DateField()),
                ('ordinal_date', models.IntegerField()),
                ('total_confirmed', models.IntegerField()),
                ('total_active', models.IntegerField()),
                ('total_recovered', models.IntegerField()),
                ('total_deaths', models.IntegerField()),
                ('total_tested', models.IntegerField()),
                ('delta_confirmed', models.IntegerField()),
                ('delta_active', models.IntegerField()),
                ('delta_recovered', models.IntegerField()),
                ('delta_deaths', models.IntegerField()),
                ('delta_tested', models.IntegerField()),
                ('delta7_confirmed', models.IntegerField()),
                ('delta7_active', models.IntegerField()),
                ('delta7_recovered', models.IntegerField()),
                ('delta7_deaths', models.IntegerField()),
                ('delta7_tested', models.IntegerField()),
                ('total_vaccinated1', models.IntegerField()),
                ('total_vaccinated2', models.IntegerField()),
                ('delta_vaccinated1', models.IntegerField()),
                ('delta_vaccinated2', models.IntegerField()),
                ('delta7_vaccinated1', models.IntegerField()),
                ('delta7_vaccinated2', models.IntegerField()),
                ('total_other', models.IntegerField()),
                ('delta_other', models.IntegerField()),
                ('delta7_other', models.IntegerField()),
            ],
        ),
        migrations.CreateModel(
            name='covid_state_data',
            fields=[
                ('index_no', models.AutoField(primary_key=True, serialize=False)),
                ('date', models.DateField()),
                ('ordinal_date', models.IntegerField()),
                ('state_name', models.CharField(max_length=255)),
                ('total_confirmed', models.IntegerField()),
                ('total_active', models.IntegerField()),
                ('total_recovered', models.IntegerField()),
                ('total_deaths', models.IntegerField()),
                ('total_tested', models.IntegerField()),
                ('delta_confirmed', models.IntegerField()),
                ('delta_active', models.IntegerField()),
                ('delta_recovered', models.IntegerField()),
                ('delta_deaths', models.IntegerField()),
                ('delta_tested', models.IntegerField()),
                ('delta7_confirmed', models.IntegerField()),
                ('delta7_active', models.IntegerField()),
                ('delta7_recovered', models.IntegerField()),
                ('delta7_deaths', models.IntegerField()),
                ('delta7_tested', models.IntegerField()),
                ('total_vaccinated1', models.IntegerField()),
                ('total_vaccinated2', models.IntegerField()),
                ('delta_vaccinated1', models.IntegerField()),
                ('delta_vaccinated2', models.IntegerField()),
                ('delta7_vaccinated1', models.IntegerField()),
                ('delta7_vaccinated2', models.IntegerField()),
                ('total_other', models.IntegerField()),
                ('delta_other', models.IntegerField()),
                ('delta7_other', models.IntegerField()),
            ],
        ),
    ]
