import joblib
model=joblib.load('./Main_Workspace/Models/Liner_regression_model_for_total_confirmed_cases_with_respect_to_date_and_total_tested.joblib')
print(model.predict([[737987,1998715]]))