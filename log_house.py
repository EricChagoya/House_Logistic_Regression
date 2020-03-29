
"""The goal is to make a logistic regression model in order to predict
whether a house will be sold within a certain threshold.
We import the data set and clean up some of the useless features.
Then we visualize the data. Then we train the model. """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


DIRECT_OLD =  "older.csv"
DIRECT_NEW = "newer.csv"

SALE_TYPE= 'SALE TYPE'
SOLD_DATE= 'SOLD DATE'
PROPERTY_TYPE= 'PROPERTY TYPE'
ADDRESS= 'ADDRESS'
CITY= 'CITY'
STATE= 'STATE OR PROVINCE'
ZIP_CODE= 'ZIP OR POSTAL CODE'
PRICE= 'PRICE'
BEDS= 'BEDS'
BATHS= 'BATHS'
LOCATION= 'LOCATION'
SQUARE_FEET= 'SQUARE FEET'
YEAR_BUILT= 'YEAR BUILT'
LOT_SIZE= 'LOT SIZE'
DAYS_ON_MARKET= 'DAYS ON MARKET'
SQFT_PER= '$/SQUARE FEET'
HOA= 'HOA/MONTH'
SOURCE= 'SOURCE'
STATUS= 'STATUS'

THRESHOLD= 180
SELLER_HOUSE= 'SELLER_HOUSE'
ACTIVE= 'Active'
SOLD= 'Sold'


def dataset()-> pd.DataFrame:
    """It returns the concated datasets as a dataframe.
    It doesn't change anything about the features."""
    old= pd.read_csv(DIRECT_OLD)
    new= pd.read_csv(DIRECT_NEW)
    return pd.concat([old, new])


def clean_up(houses:pd.DataFrame) -> pd.DataFrame:
    """Here I remove columns we will never use. Remove columns with mostly
    missing data. Add a binary value for Seller_House.
    I change the column names so they don't have underscores bc the spaces
    messes up the logistic regression."""
    houses= delete_columns(houses)
    houses= analyze_missing_values(houses)
    houses= add_seller_house(houses)
    houses= add_underscore(houses)
    houses= create_dummies(houses)
    houses= impute(houses)
    return houses


def delete_columns(houses:pd.DataFrame)-> pd.DataFrame:
    """No use for these columns because they are mostly empty, all share the same
    attributes, or the column will cause overfitting."""
    drop_columns= ['NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME', 
                   'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)',
                   'MLS#', 'FAVORITE', 'INTERESTED', 'LATITUDE', 'LONGITUDE',
                   SOURCE, SALE_TYPE, CITY, STATE]
    houses= houses[houses[STATUS].isin(['Sold'])]
    houses= houses[houses[CITY].isin(['Irvine'])]
    return houses.drop(drop_columns, axis= 1)


def analyze_missing_values(houses:pd.DataFrame) -> pd.DataFrame:
    """It looks at where there a large number of missing values."""
    drop_columns= [SOLD_DATE, LOT_SIZE]
    houses= houses.drop(drop_columns, axis= 1)
    #print(houses.describe(include= 'all'))
    #print(houses.isna().sum())
    #sns.heatmap(houses.isnull(),yticklabels=False,cbar=False, cmap='viridis')
    #plt.show()
    
    # Too many missing value for columns of Sold Date, Lot Size
    # About a quarter of the buy houses have missing values for lot sizes. If it's missing, 
    # it's either because the lot size is 0 or they are not reporting it.
    # There is a lot of missing values for the sold houses. Usually when a house gets
    # sold, a lot of the information is removed from the website. From one of the
    # houses that were on the buy dataset got sold within the past two days and some
    # of the information got removed like the price sold, beds, baths, sq ft, etc.
    return houses


def add_seller_house(houses:pd.DataFrame) -> pd.DataFrame:
    """If the days on market is 180 or above, then it's a seller's house.
    If the days on market is less than 180 it is a buyer's house."""
    houses[SELLER_HOUSE]= 2
    above= houses[DAYS_ON_MARKET] >= THRESHOLD
    below= (houses[DAYS_ON_MARKET] < THRESHOLD)
    sold= (houses[STATUS] == SOLD)
    s1= [x and y for x,y in zip(above, sold)]
    s2= [x and y for x,y in zip(below, sold)]
    houses.loc[s1, SELLER_HOUSE]= 1
    houses.loc[s2, SELLER_HOUSE]= 0
    return houses[houses.SELLER_HOUSE != 2]


def add_underscore(houses:pd.DataFrame) -> pd.DataFrame:
    """I was having problems accessing the variables in the model phase when there
    was spaces in the columns."""
    houses['PROPERTY_TYPE']= houses[PROPERTY_TYPE]
    houses['ZIP_CODE']= houses[ZIP_CODE]
    houses['SQFT']= houses[SQUARE_FEET]
    houses['YEAR_BUILT']= houses[YEAR_BUILT]
    houses['DAYS_ON_MARKET']= houses[DAYS_ON_MARKET]
    houses['SQFT_PER']= houses[SQFT_PER]
    houses['HOA']= houses[HOA]
    drop_columns= [PROPERTY_TYPE, ZIP_CODE, SQUARE_FEET, YEAR_BUILT, 
                   DAYS_ON_MARKET, SQFT_PER, HOA]
    return houses.drop(drop_columns, axis= 1)


def create_dummies(houses:pd.DataFrame) -> pd.DataFrame:
    """Create dummy variables for categorical data. Location, Zip Code, Property Type. """
    #houses= houses.rename(columns= {'PROPERTY_TYPE':'PT', 'ZIP_CODE':'ZIP', 'LOCATION':'LOC'})
    houses['PT']= houses['PROPERTY_TYPE']
    houses['ZIP']= houses['ZIP_CODE']
    houses['LOC']= houses['LOCATION']
    return pd.get_dummies(houses, columns= ['PT', 'ZIP', 'LOC'])


def impute(houses:pd.DataFrame) -> pd.DataFrame:
    single= (houses['PROPERTY_TYPE'] == 'Single Family Residential')
    houses['HOA'].fillna(houses['HOA'][single].mean(), inplace= True)
    houses['BATHS'].fillna(round(houses['BATHS'].mean()), inplace= True)
    return houses


def visualize(houses:pd.DataFrame) -> None:
    """Visualizes the data to see if there are any outliers or anything special."""
    #price_distribution(houses)
    #prop_types(houses)
    #zip_code(houses)
    #year_built(houses)
    #bed_bath(houses)
    return


def price_distribution(houses:pd.DataFrame) -> None:
    """The first is a histogram of house prices. I removed houses that
    sold for more than 3000000 bc it was difficult to see the spread.
    The second is a scatter plot of Price and Day on the Market."""
    """
    indexNames= houses[houses[PRICE] >= 3000000].index
    print('Mean: ', houses[PRICE].mean().round())
    print('Median: ', houses[PRICE].median().round())
    """
    indexNames= houses[houses['PRICE'] >= 3000000].index
    houses= houses.drop(indexNames)
    plt.hist(houses['PRICE'])
    plt.xlabel('Price in Dollars')
    plt.ylabel('Count')
    plt.show()

    cond= houses[STATUS] == ACTIVE
    sold_houses= houses[~cond]
    plt.scatter(sold_houses['PRICE'], sold_houses['DAYS_ON_MARKET'], s= 5)
    plt.legend()
    plt.xlabel('Price in Dollars')
    plt.ylabel('Days on the Market')
    plt.show()


def prop_types(houses:pd.DataFrame) -> None:
    """The first is a box and whisker plot of different property type.
    The second is a bar graph of the counts of each property type."""
    sns.set_style('whitegrid')
    indexNames= houses[houses['PRICE'] >= 3000000].index
    houses= houses.drop(indexNames)
    
    ax= sns.catplot(x= 'PROPERTY_TYPE', y= 'PRICE', kind= 'box', data= houses)
    ax.set_xticklabels(rotation=30)
    plt.tight_layout()
    plt.show()
    
    ax= sns.countplot(x= 'PROPERTY_TYPE', data= houses)
    ax.set_xticklabels(ax.get_xticklabels(), rotation= 30, ha="right", fontsize=9)
    plt.show() 


def zip_code(houses:pd.DataFrame) -> None:
    """First shows a box and whisker plot of price and zip code.
    The second is bar graph of zip code count."""
    indexNames= houses[houses['PRICE'] >= 3000000].index
    houses= houses.drop(indexNames)
    ax= sns.catplot(x= 'ZIP_CODE', y= 'PRICE', kind= 'box', data= houses)
    ax.set_xticklabels(rotation=30)
    plt.tight_layout()
    plt.show()
    
    ax1= sns.countplot(x= 'ZIP_CODE', data= houses)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation= 30, ha="right", fontsize=9)
    plt.show()
    #Zip code looks kinda important. Some areas only have a few houses. Some areas
    #do look they are in a better part of town based off price.


def year_built(houses:pd.DataFrame) -> None:
    """The first is a bar chart of counts of Year count.
    The second is a scatterplot of of Price and Year Built."""
    plt.hist(houses['YEAR_BUILT'])
    plt.xlabel('Year Built')
    plt.ylabel('Count')
    plt.show()
    
    indexNames= houses[houses['PRICE'] >= 3000000].index
    houses= houses.drop(indexNames)    
    plt.scatter(houses['YEAR_BUILT'], houses['PRICE'])
    plt.xlabel('Year Built')
    plt.ylabel('Price in Dollars')
    plt.show()


def bed_bath(houses:pd.DataFrame) -> None:
    """A scatterplot of Beds and Bathrooms with property 
    type being shown in color"""
    houses= houses.dropna(subset= ['BEDS', 'BATHS', 'PROPERTY_TYPE'])
    sns.catplot(x= 'BEDS', y= 'BATHS', hue= 'PROPERTY_TYPE', data= houses)
    plt.show()


def analysis(houses:pd.DataFrame) -> None:
    """We remove columns that were replaced by dummy variables
    and others just used for visualization.
    We use RFE to create a model and test how that works out. 
    We then create another model based off the variables were statistically significant.
    """
    
    """
    #Me just trying to fit the data without any outside influences
    f= f'SELLER_HOUSE ~ SQFT_PER + PRICE + C(LOCATION)'    
    result= smf.logit(formula= str(f), data= houses).fit()
    print(result.summary2())
    y= ['SELLER_HOUSE']
    x= ['SQFT_PER', 'PRICE', 'LOC_699 - Not Defined', 'LOC_AA - Airport Area', 'LOC_CG - Columbus Grove',
       'LOC_CV - Cypress Village', 'LOC_EASTW - Eastwood', 'LOC_EC - El Camino Real', 'LOC_GP - Great Park',
       'LOC_IRSP - Irvine Spectrum', 'LOC_LGA - Laguna Altura', 'LOC_NK - Northpark', 'LOC_NW - Northwood', 
        'LOC_OC - Oak Creek', 'LOC_OH - Orchard Hills', 'LOC_OT - Orangetree', 'LOC_PS - Portola Springs', 
        'LOC_QH - Quail Hill', 'LOC_SH - Shady Canyon', 'LOC_SJ - Rancho San Joaquin', 'LOC_STG - Stonegate', 
        'LOC_Stonegate', 'LOC_TR - Turtle Rock', 'LOC_TRG - Turtle Ridge', 'LOC_UP - University Park',
       'LOC_UT - University Town Center', 'LOC_WB - Woodbridge', 'LOC_WD - Woodbury', 
        'LOC_WI - West Irvine', 'LOC_WN - Walnut (Irvine)', 'LOC_WP - Westpark']
    x_train, x_test, y_train, y_test= train_test_split(houses[x], houses[y], test_size= 0.3, random_state= 500)
    logreg = LogisticRegression()
    logreg.fit(x_train, y_train.values.ravel())
    y_pred= logreg.predict(x_test)
    print('Accuracy of logistic regression classifier on test set:', round(logreg.score(x_test, y_test), 3))
    # This model is really bad
    
    """
    
    ""
    houses= houses.drop(['DAYS_ON_MARKET', 'ADDRESS', 'LOCATION',
                         'STATUS', 'PROPERTY_TYPE', 'ZIP_CODE'], axis= 1)
    columns= houses.columns.values.tolist()
    y= ['SELLER_HOUSE']
    x= [i for i in columns if i not in y]
    
    # Over Sampling Using SMOTE 
    x_train, _, y_train, _= train_test_split(houses[x], houses[y], test_size= 0.3, random_state= 500)
    x_columns= x_train.columns
    
    os= SMOTE(random_state= 0)
    os_x, os_y= os.fit_sample(x_train, y_train)
    os_x= pd.DataFrame(data= os_x, columns= x_columns)
    os_y= pd.DataFrame(data= os_y, columns= y)
    
    
    #Recursive Feature Elimination
    logreg= LogisticRegression(max_iter= 600)
    rfe= RFE(logreg, 20)
    rfe= rfe.fit(os_x, os_y.values.ravel())
    
    lst= [i for count, i in enumerate(x) if rfe.support_[count] == True]
    X= os_x[lst]
    Y= os_y['SELLER_HOUSE']
    
    
    #logit_model= sm.Logit(Y, X)
    #result= logit_model.fit()
    #print(result.summary2())    # Model choosen by RCE
    
    #These are features have a p-value less than 0.05
    final_x= ['BATHS',  'ZIP_92602.0', 'ZIP_92618.0', 'LOC_699 - Not Defined', 'LOC_TR - Turtle Rock', 'LOC_WD - Woodbury']
    #final_x= ['ZIP_92602.0', 'LOC_699 - Not Defined', 'LOC_TR - Turtle Rock', 'LOC_WD - Woodbury']
    X2= os_x[final_x]
    
    logit_model2= sm.Logit(Y, X2)
    result2= logit_model2.fit()
    print(result2.summary2())   # Final Model
    
    x_train2, x_test2, y_train2, y_test2= train_test_split(X2, Y, test_size= 0.3, random_state= 500)
    logreg = LogisticRegression()
    logreg.fit(x_train2, y_train2)
    
    y_pred= logreg.predict(x_test2)
    print('Accuracy of logistic regression classifier on test set:', round(logreg.score(x_test2, y_test2), 2))
    
    conf_matrix= confusion_matrix(y_test2, y_pred)
    print(conf_matrix)
    # So 22+61 correct predictions and 13+44 wrong predictions
    
    logit_roc_auc = roc_auc_score(y_test2, logreg.predict(x_test2))
    fpr, tpr, _ = roc_curve(y_test2, logreg.predict_proba(x_test2)[:,1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    ""


def main() -> None:
    pd.set_option("display.max_columns", 40)
    houses= dataset()
    houses= clean_up(houses)
    visualize(houses)
    analysis(houses)


if __name__== '__main__':
    main()

