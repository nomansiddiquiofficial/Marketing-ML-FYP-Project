import os
from django.http import HttpResponse
from django.shortcuts import render

import pandas as pd
import json




from sklearn.metrics import confusion_matrix


# Create your views here.
# def loaddata():
  
#     df = pd.read_excel("c:\\Users\\Noman Siddiqui\\Downloads\\a2_Dataset_90Percent.xlsx",)
#     print(df)
#     for _, row in df.iterrows():
#         person = Person(DemReg=row['DemReg'], LoyalClass=row['LoyalClass'])
#         person.save()



def about(request):
    #get_dataframe()
    return render(request, 'about.html' , {"title":"About"})


# 10_percent df connection
 
sheet_id = "1m77YwMVIiPmSZPzXuIhhQjIgrhIidnDC"
sheet_name = "Sheet1"

url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
url = url.replace(" ", "%20")


def get_org_dataframe(request):

    df = pd.read_csv(url)                                                                           
    df = df.head(20)

    json_records = df.to_json(orient='records')
    arr = []
    arr = json.loads(json_records)

    context = {'data': arr}

    print(json_records)
    print(arr)

    return render(request, 'dataframe.html' , context)



def get_malupilated_dataframe():
    df = pd.read_csv(url)

    #filling missing values with mean/mode
    df['DemAffl'] = df['DemAffl'].fillna(df['DemAffl'].mode()[0])
    df['DemAge'] = df['DemAge'].fillna(df['DemAge'].mode()[0])
    df['DemClusterGroup'] = df['DemClusterGroup'].fillna(df['DemClusterGroup'].mode()[0])
    df['DemGender'] = df['DemGender'].fillna(df['DemGender'].mode()[0])
    df['DemReg'] = df['DemReg'].fillna(df['DemReg'].mode()[0])
    df['DemTVReg'] = df['DemTVReg'].fillna(df['DemTVReg'].mode()[0])
    df['LoyalTime'] = df['LoyalTime'].fillna(df['LoyalTime'].mean())
    
    return df

from statsmodels.stats.outliers_influence import variance_inflation_factor
 # Calculate VIF
def calc_vif(z):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = z.columns
    vif["VIF"] = [variance_inflation_factor(z.values, i) for i in range(z.shape[1])]
    return vif

from sklearn.preprocessing import LabelEncoder

def combined_dataframe_view(request):

    # Fetch the original DataFrame
    df_original = get_malupilated_dataframe()
    df_original = df_original.head(20)
    
    # Convert the original DataFrame to JSON
    json_records_original = df_original.to_json(orient='records')
    arr_original = json.loads(json_records_original)

    # Fetch and convert the DataFrame
    df_converted = df_original.copy()
    number = LabelEncoder()

    #Coverting category to numeric
    df_converted['DemClusterGroup'] = number.fit_transform(df_converted['DemClusterGroup'].astype('str'))
    df_converted['DemGender'] = number.fit_transform(df_converted['DemGender'].astype('str'))
    df_converted['DemReg'] = number.fit_transform(df_converted['DemReg'].astype('str'))
    df_converted['DemTVReg'] = number.fit_transform(df_converted['DemTVReg'].astype('str'))
    df_converted['LoyalClass'] = number.fit_transform(df_converted['LoyalClass'].astype('str'))
    
    df_converted = df_converted.head(20)

    # Convert the converted DataFrame to JSON
    json_records_converted = df_converted.to_json(orient='records')
    arr_converted = json.loads(json_records_converted)

    # Context for the template
    context = {
        'data_original': arr_original,
        'data_converted': arr_converted
    }

    # Calculate VIF
    # z = df_converted.iloc[:, 0:9]
    # vif = calc_vif(z)
    # print("VIF results:\n", vif)

    return render(request, 'mal_dataframe.html', context)


from sklearn.model_selection import train_test_split

def model_view(request):
   
    return render(request, 'modelling.html', )



from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import accuracy_score 
import joblib

decile_counts = {}
prob_threshold = {}
good_total = {}
good_total_percentage = {}
bad_total = {}
bad_total_percentage = {}
good_cummulative = {}
good_cummulative_percentage = {}
bad_cummulative = {}
bad_cummulative_percentage = {}
profits = {}

dic_for_insights = {}


def train_model_view(request):
    global decile_counts,prob_threshold,good_total,good_total_percentage,bad_total,bad_total_percentage,good_cummulative,good_cummulative_percentage,bad_cummulative,bad_cummulative_percentage,profits

    decile_counts = {}
    prob_threshold = {}
    good_total = {}
    good_total_percentage = {}
    bad_total = {}
    bad_total_percentage = {}
    good_cummulative = {}
    good_cummulative_percentage = {}
    bad_cummulative = {}
    bad_cummulative_percentage = {}
    profits = {}

    dic_for_insights = {}
   
   
    number = LabelEncoder()
    df = get_malupilated_dataframe()

    df['DemClusterGroup'] = number.fit_transform(df['DemClusterGroup'].astype('str'))
    df['DemGender'] = number.fit_transform(df['DemGender'].astype('str'))
    df['DemReg'] = number.fit_transform(df['DemReg'].astype('str'))
    df['DemTVReg'] = number.fit_transform(df['DemTVReg'].astype('str'))
    df['LoyalClass'] = number.fit_transform(df['LoyalClass'].astype('str'))

    y = df.iloc[:, 10].values
    X = df.iloc[:, 0:10].values

   
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=0)

    # Modelling 
    classifier = LogisticRegression(max_iter=200)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)


    # # Exporting Logistic Regression Classifier to later use in prediction
   
    folder_path = 'S:/django_project/marketing_ml_fyp/ml_models'
    os.makedirs(folder_path, exist_ok=True)
    model_path = os.path.join(folder_path, '3.2_Classifier_LoyalCustomers.pkl')

    # # Exporting Logistic Regression Classifier to later use in prediction
    joblib.dump(classifier, model_path)

    # print(confusion_matrix(y_test, y_pred))

    # print(accuracy_score(y_test, y_pred))

    predictions = classifier.predict_proba(X_test)
 
    # Writing model output file
    df_prediction_prob = pd.DataFrame(predictions, columns=['prob_0', 'prob_1'])
    df_test_dataset = pd.DataFrame(y_test, columns=['ActualOutcome'])
    df_x_test = pd.DataFrame(X_test)

    # Concatenate dataframes
    dfx = pd.concat([df_x_test, df_test_dataset, df_prediction_prob], axis=1)

     # Sort the DataFrame by 'prob_2' in descending order
    dfx = dfx.sort_values(by='prob_1', ascending=False)

    # Create the Decile column
    dfx['Decile'] = 10 - pd.qcut(dfx['prob_1'], 10, labels=False)
 
    dfx.to_excel("ModelOutput_10Percent.xlsx", index=False)
        
   

    file_path = r"S:\django_project\marketing_ml_fyp\marketing_ml_project\ModelOutput_10Percent.xlsx"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} ko delete kar diya gaya hai.")
        dfx.to_excel("ModelOutput_10Percent.xlsx", index=False)
    else:
        print(f"{file_path} exist nahi karti.")
        # Save the DataFrame to an Excel file

   
    df = pd.read_excel(file_path)
    
    
   
    cumulative_good_count = 0
    cumulative_bad_count = 0
    

    # Loop through each decile from 1 to 10
    for i in range(1,11):
    # col 1 -->  Count the number of occurrences of the current decile
        each_decile_count = df[df['Decile'] == i].shape[0]      #.shape[0] to count the number of rows that meet your conditions.

    # col 2 -->
        positive_count = df[(df['ActualOutcome'] == 0) & (df['Decile'] == i)].shape[0]
        prob_thres = round((positive_count / each_decile_count) * 100, 2) 

    # col 3 -->
        good_count = df[(df['ActualOutcome'] == 1) & (df['Decile'] == i)].shape[0]
        good_total_count =  good_count  
       
        
    # col 4 -->
        positive_count = df[(df['ActualOutcome'] == 1) & (df['Decile'] == i)].shape[0]
        good_count_percentage = round((positive_count / each_decile_count) * 100, 2) 
            
    # col 5 -->
        bad_count = df[(df['ActualOutcome'] == 0) & (df['Decile'] == i)].shape[0]
        bad_total_count =  bad_count  
        
    # col 6 -->
        bad_count = df[(df['ActualOutcome'] == 0) & (df['Decile'] == i)].shape[0]
        bad_count_percentage = round((bad_count / each_decile_count) * 100, 2) 

    # col 7 -->
        good_count = df[(df['ActualOutcome'] == 1) & (df['Decile'] == i)].shape[0]
        cumulative_good_count += good_count
        
    # col 8 -->
        good_total_rows = df[df['ActualOutcome'] == 1].shape[0]
        cumulative_good_percentage = round((cumulative_good_count / good_total_rows) * 100,2)

     # col 9 -->
        bad_count = df[(df['ActualOutcome'] == 0) & (df['Decile'] == i)].shape[0]
        cumulative_bad_count += bad_count
    
    # col 10 -->
        bad_total_rows = df[df['ActualOutcome'] == 0].shape[0]
        cumulative_bad_percentage = round((cumulative_bad_count / bad_total_rows) * 100,2)

    
       
        
        # Add the count to the dictionary with the appropriate key
        decile_counts[f'total{i}'] = each_decile_count
        prob_threshold[f'prob_thres{i}'] = prob_thres
        good_total[f'good_total{i}'] = good_total_count
        good_total_percentage[f'good_total_per{i}'] = good_count_percentage
        bad_total[f'bad_total{i}'] = bad_total_count
        bad_total_percentage[f'bad_total_per{i}'] = bad_count_percentage
        good_cummulative[f'good_cumm{i}'] = cumulative_good_count
        good_cummulative_percentage[f'good_cumm_per{i}'] = cumulative_good_percentage
        bad_cummulative[f'bad_cumm{i}'] = cumulative_bad_count
        bad_cummulative_percentage[f'bad_cumm_per{i}'] = cumulative_bad_percentage
        
    for i in range(1,11):    
        # col 11 (profit) -->
        revenue_of_buyer = request.POST.get('revenue_per_buyer')
        cost_pro_kit = request.POST.get('cost_per_kit')

        if revenue_of_buyer and cost_pro_kit:
        
            revenue_of_buyer = float(revenue_of_buyer)
            cost_pro_kit = float(cost_pro_kit)

            dic_key = f'good_cumm{i}'
            dic_key1 = f'bad_cumm{i}'
            profit = (revenue_of_buyer * good_cummulative[dic_key])-(cost_pro_kit*(good_cummulative[dic_key]+bad_cummulative[dic_key1]))
            profits[f'profit{i}'] = profit
        else: 
            revenue_of_buyer = 0 
            cost_pro_kit = 0
            dic_key = f'good_cumm{i}'
            dic_key1 = f'bad_cumm{i}'
            profit = (revenue_of_buyer * good_cummulative[dic_key])-(cost_pro_kit*(good_cummulative[dic_key]+bad_cummulative[dic_key1]))
            profits[f'profit{i}'] = profit
       
        
    json_records = df.to_json(orient='records')
    arr = []
    arr = json.loads(json_records)


    dic_for_insights = {'data': arr}

    dic_for_insights.update(decile_counts)
    dic_for_insights.update(prob_threshold)
    dic_for_insights.update(good_total)
    dic_for_insights.update(good_total_percentage)
    dic_for_insights.update(bad_total)
    dic_for_insights.update(bad_total_percentage)
    dic_for_insights.update(good_cummulative)  
    dic_for_insights.update(good_cummulative_percentage)    
    dic_for_insights.update(bad_cummulative)  
    dic_for_insights.update(bad_cummulative_percentage)   
    dic_for_insights.update(profits) 
    print(decile_counts)
    print(profits)
   
    return render(request, 'train_model.html',dic_for_insights)

import plotly.express as px
 
def graphing(request):
    
    data = {
        'Decile': [decile_counts['total1'] , decile_counts['total2'], decile_counts['total3'],decile_counts['total4'] , decile_counts['total5'], decile_counts['total6'],decile_counts['total7'] , decile_counts['total8'], decile_counts['total9'],decile_counts['total10']],
        'Profit (In PKR)': [profits['profit1'],profits['profit2'],profits['profit3'],profits['profit4'],profits['profit5'],profits['profit6'],profits['profit7'],profits['profit8'],profits['profit9'],profits['profit10']]
    }
    fig = px.bar(data, x='Decile', y='Profit (In PKR)', title='Profit across different Deciles')
    graph = fig.to_html(full_html=False)

    data = {
        'Decile': [
            decile_counts['total1'] , decile_counts['total2'], decile_counts['total3'],decile_counts['total4'] , decile_counts['total5'], decile_counts['total6'],decile_counts['total7'] , decile_counts['total8'], decile_counts['total9'],decile_counts['total10']
            ],
        'Cumulative Good': [
            good_cummulative['good_cumm1'], good_cummulative['good_cumm2'], good_cummulative['good_cumm3'], 
            good_cummulative['good_cumm4'], good_cummulative['good_cumm5'], good_cummulative['good_cumm6'], 
            good_cummulative['good_cumm7'], good_cummulative['good_cumm8'], good_cummulative['good_cumm9'], 
            good_cummulative['good_cumm10']
        ],
        'Cumulative Bad': [
            bad_cummulative['bad_cumm1'], bad_cummulative['bad_cumm2'], bad_cummulative['bad_cumm3'], 
            bad_cummulative['bad_cumm4'], bad_cummulative['bad_cumm5'], bad_cummulative['bad_cumm6'], 
            bad_cummulative['bad_cumm7'], bad_cummulative['bad_cumm8'], bad_cummulative['bad_cumm9'], 
            bad_cummulative['bad_cumm10']
        ]
    }
    

    fig = px.bar(data, x='Decile', y=['Cumulative Good', 'Cumulative Bad'], 
                  title='Cumulative Distribution of Good and Bad Outcomes',
                  labels={'value': 'Cumulative Count', 'variable': 'Outcome'})
  
    graph1 = fig.to_html(full_html=False)

    # Calculate the total counts from the dictionaries
    total_good = sum(good_total.values())
    total_bad = sum(bad_total.values())

   
    data = {
        'TargetBuy': ['Good (Target Buyers)', 'Bad (Non-Target Buyers)'],
        'Count': [total_good, total_bad]
    }

    # Create the pie chart
    fig = px.pie(data, names='TargetBuy', values='Count',
                 title='Proportion of Target Buyers vs Non-Target Buyers')

    # Convert the Plotly figure to HTML for embedding in Django template
    graph2 = fig.to_html(full_html=False)


  
    df = pd.read_csv(url)  
    
    # Create a scatter plot
    fig = px.scatter(df, x=df['LoyalSpend'], y=df['LoyalTime'],
                     title='Relationship between LoyalSpend and LoyalTime',
                     labels={'LoyalSpend': 'Loyal Spend (Amount)', 'LoyalTime': 'Loyal Time (Months)'})

   
    graph3 = fig.to_html(full_html=False)
                                                                                                
    # Create a box plot
    fig = px.box(df, x=df['LoyalClass'], y=df['LoyalSpend'],
                 title='Spending Habits Across Different Loyalty Classes',
                 labels={'LoyalClass': 'Loyalty Class', 'LoyalSpend': 'Loyal Spend (Amount)'})

    # Convert the Plotly figure to HTML for embedding in Django template
    graph4 = fig.to_html(full_html=False)

    return render(request, 'graphs.html', {'graph': graph, 'graph1':graph1, 'graph2':graph2, 'graph3':graph3, 'graph4':graph4})




def predict_model_view(request):
    sheet_id = "1LqcmXuzcms9ZWrqaSXKHrJHLxN4ANUbq"
    sheet_name = "Sheet1"

    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    url = url.replace(" ", "%20")
    df = pd.read_csv(url)

    #filling missing values with mean/mode
    df['DemAffl'] = df['DemAffl'].fillna(df['DemAffl'].mode()[0])
    df['DemAge'] = df['DemAge'].fillna(df['DemAge'].mode()[0])
    df['DemClusterGroup'] = df['DemClusterGroup'].fillna(df['DemClusterGroup'].mode()[0])
    df['DemGender'] = df['DemGender'].fillna(df['DemGender'].mode()[0])
    df['DemReg'] = df['DemReg'].fillna(df['DemReg'].mode()[0])
    df['DemTVReg'] = df['DemTVReg'].fillna(df['DemTVReg'].mode()[0])
    df['LoyalTime'] = df['LoyalTime'].fillna(df['LoyalTime'].mean())


    number = LabelEncoder()

    #Coverting category to numeric
    df['DemClusterGroup'] = number.fit_transform(df['DemClusterGroup'].astype('str'))
    df['DemGender'] = number.fit_transform(df['DemGender'].astype('str'))
    df['DemReg'] = number.fit_transform(df['DemReg'].astype('str'))
    df['DemTVReg'] = number.fit_transform(df['DemTVReg'].astype('str'))
    df['LoyalClass'] = number.fit_transform(df['LoyalClass'].astype('str'))

    #predictions
    X_fresh = df.iloc[:,0:10].values
    folder_path = 'S:/django_project/marketing_ml_fyp/ml_models/3.2_Classifier_LoyalCustomers.pkl'
    classifier = joblib.load(folder_path)

    y_pred = classifier.predict(X_fresh)
    print(y_pred)

    predictions = classifier.predict_proba(X_fresh)
    
    # writing model output file
    df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])

    df = pd.concat([df, df_prediction_prob], axis=1)

    df.to_excel("BuyProb_90Percent.xlsx")

    json_records = df.to_json(orient='records')
    arr = []
    arr = json.loads(json_records)

    df = pd.read_excel('S:\django_project\marketing_ml_fyp\marketing_ml_project\BuyProb_90Percent.xlsx')
    fig = px.scatter(
            df,
            x=df['LoyalSpend'],
            y=df['prob_0'],
            color='LoyalSpend',  
            title='Relationship between Loyal Spend and Probability of Being a Targeted Buyer',
            labels={
                'LoyalSpend': 'Loyal Spend',
                'Probability': 'Probability (prob_1)'
            }
        )
    graph = fig.to_html(full_html=False)

    df_avg_prob = df.groupby('DemTVReg')['prob_1'].mean().reset_index()

    # Create the bar chart
    fig = px.bar(df_avg_prob, x=df['DemTVReg'], y=df['prob_1'],
                title='Impact of TV Region on Probability of Being a Targeted Buyer',
                labels={'DemTVReg': 'TV Region', 'prob_1': 'Average Probability of Targeted Buyer'})

    # Render the graph in HTML
    graph1 = fig.to_html(full_html=False)


    df = df.sort_values(by='LoyalTime')

    # Calculate the cumulative sum of prob_1
    df['cumulative_prob_1'] = df['prob_1'].cumsum()

    # Create the cumulative line chart
    fig = px.line(df, x=df['LoyalTime'], y=df['cumulative_prob_1'], 
                title='Cumulative Probability over Time',
                labels={'LoyalTime': 'Loyalty Time', 'cumulative_prob_1': 'Cumulative Probability of Being a Targeted Buyer'})

    # Render the graph in HTML
    graph2 = fig.to_html(full_html=False)



    fig = px.density_heatmap(df, x='DemAge', y='LoyalSpend', z='prob_1', 
                         title='Heatmap of Age and Spending vs. Probability of Being a Targeted Buyer',
                         labels={'DemAge': 'Age', 'LoyalSpend': 'Spending', 'prob_1': 'Probability'},
                         nbinsx=20, nbinsy=20, color_continuous_scale='Viridis')

    # Render the graph in HTML
    graph3 = fig.to_html(full_html=False)

    dic_for_predict = {'data': arr, 'graph': graph, 'graph1': graph1, 'graph2':graph2, 'graph3':graph3}

    return render(request, 'predict_model.html',dic_for_predict)



