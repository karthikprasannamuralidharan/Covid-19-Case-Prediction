# Generated by Django 3.2.7 on 2021-10-02 09:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('covid', '0002_delete_covid_district_data'),
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
                ('delta_confirmed', models.IntegerField()),
                ('delta_active', models.IntegerField()),
                ('delta_recovered', models.IntegerField()),
                ('delta_deaths', models.IntegerField()),
                ('delta7_confirmed', models.IntegerField()),
                ('delta7_active', models.IntegerField()),
                ('delta7_recovered', models.IntegerField()),
                ('delta7_deaths', models.IntegerField()),
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
