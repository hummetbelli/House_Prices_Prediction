# Gerekli import işlemleri gerçekleştirildi

import lightgbm
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost
from lightgbm import LGBMClassifier
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

from warnings import filterwarnings
filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Data fields

#Here's a brief version of what you'll find in the data description file.
#SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
#MSSubClass: The building class
#MSZoning: The general zoning classification
#LotFrontage: Linear feet of street connected to property
#LotArea: Lot size in square feet
#Street: Type of road access
#Alley: Type of alley access
#LotShape: General shape of property
#LandContour: Flatness of the property
#Utilities: Type of utilities available
#LotConfig: Lot configuration
#LandSlope: Slope of property
#Neighborhood: Physical locations within Ames city limits
#Condition1: Proximity to main road or railroad
#Condition2: Proximity to main road or railroad (if a second is present)
#BldgType: Type of dwelling
#HouseStyle: Style of dwelling
#OverallQual: Overall material and finish quality
#OverallCond: Overall condition rating
#YearBuilt: Original construction date
#YearRemodAdd: Remodel date
#RoofStyle: Type of roof
#RoofMatl: Roof material
#Exterior1st: Exterior covering on house
#Exterior2nd: Exterior covering on house (if more than one material)
#MasVnrType: Masonry veneer type
#MasVnrArea: Masonry veneer area in square feet
#ExterQual: Exterior material quality
#ExterCond: Present condition of the material on the exterior
#Foundation: Type of foundation
#BsmtQual: Height of the basement
#BsmtCond: General condition of the basement
#BsmtExposure: Walkout or garden level basement walls
#BsmtFinType1: Quality of basement finished area
#BsmtFinSF1: Type 1 finished square feet
#BsmtFinType2: Quality of second finished area (if present)
#BsmtFinSF2: Type 2 finished square feet
#BsmtUnfSF: Unfinished square feet of basement area
#TotalBsmtSF: Total square feet of basement area
#Heating: Type of heating
#HeatingQC: Heating quality and condition
#CentralAir: Central air conditioning
#Electrical: Electrical system
#1stFlrSF: First Floor square feet
#2ndFlrSF: Second floor square feet
#LowQualFinSF: Low quality finished square feet (all floors)
#GrLivArea: Above grade (ground) living area square feet
#BsmtFullBath: Basement full bathrooms
#BsmtHalfBath: Basement half bathrooms
#FullBath: Full bathrooms above grade
#HalfBath: Half baths above grade
#Bedroom: Number of bedrooms above basement level
#Kitchen: Number of kitchens
#KitchenQual: Kitchen quality
#TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#Functional: Home functionality rating
#Fireplaces: Number of fireplaces
#FireplaceQu: Fireplace quality
#GarageType: Garage location
#GarageYrBlt: Year garage was built
#GarageFinish: Interior finish of the garage
#GarageCars: Size of garage in car capacity
#GarageArea: Size of garage in square feet
#GarageQual: Garage quality
#GarageCond: Garage condition
#PavedDrive: Paved driveway
#WoodDeckSF: Wood deck area in square feet
#OpenPorchSF: Open porch area in square feet
#EnclosedPorch: Enclosed porch area in square feet
#3SsnPorch: Three season porch area in square feet
#ScreenPorch: Screen porch area in square feet
#PoolArea: Pool area in square feet
#PoolQC: Pool quality
#Fence: Fence quality
#MiscFeature: Miscellaneous feature not covered in other categories
#MiscVal: $Value of miscellaneous feature
#MoSold: Month Sold
#YrSold: Year Sold
#SaleType: Type of sale
#SaleCondition: Condition of sale

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train["PoolQC"].unique()
train.head()
test.head()
train.info()
test.info()



print(f"Number of rows and number of columns in the train dataset are {train.shape[0]} and {train.shape[1]}")
print(f"Number of rows and number of columns in the test dataset are {test.shape[0]} and {test.shape[1]}")

#Finding Null Values

for col in train:
    if train[col].isnull().sum() != 0:
        print(f'{col} : {train[col].isnull().sum()} null values | Dtype : {train[col].dtype}')

for col in test:
    if test[col].isnull().sum() != 0:
        print(f'{col} : {test[col].isnull().sum()} null values | Dtype : {test[col].dtype}')

total_train = train.isnull().sum().sort_values(ascending=False)
total_select = total_train.head(20)
total_select.plot(kind="bar", figsize=(8, 6), fontsize=10)

plt.xlabel("Columns", fontsize=20)
plt.ylabel("Total Count", fontsize=20)
plt.title("Total Missing Values Train Data", fontsize=20)
plt.show()

total_test = test.isnull().sum().sort_values(ascending=False)
total_select = total_test.head(20)
total_select.plot(kind="bar", figsize=(8, 6), fontsize=10)

plt.xlabel("Columns", fontsize=20)
plt.ylabel("Total Count", fontsize=20)
plt.title("Total Missing Values Test Data", fontsize=20)
plt.show()


#Visualizations Of Variables

train.describe().T
test.describe().T

print("Basic descriptive statistics of the target variable - 'SalePrice': \n\n",
      train["SalePrice"].describe())

train.hist(figsize = (30, 30), bins = 20, legend = False)
plt.show()

var = 'OverallQual'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show()

df_main = pd.concat([train,test],ignore_index=True)


# Alley : data description says NA means "no alley access"
# Ev müstakil mi onu ifade ediyor olabilir.
df_main.loc[:, "Alley"] = df_main.loc[:, "Alley"].fillna("None")
# BedroomAbvGr : NA most likely means 0
df_main.loc[:, "BedroomAbvGr"] = df_main.loc[:, "BedroomAbvGr"].fillna(0)
# BsmtQual etc : data description says NA for basement features is "no basement"
df_main.loc[:, "BsmtQual"] = df_main.loc[:, "BsmtQual"].fillna("No")
df_main.loc[:, "BsmtCond"] = df_main.loc[:, "BsmtCond"].fillna("No")
df_main.loc[:, "BsmtExposure"] = df_main.loc[:, "BsmtExposure"].fillna("No")
df_main.loc[:, "MSZoning"] = df_main.loc[:, "MSZoning"].mode()[0]
df_main.loc[:, "Exterior1st"] = df_main.loc[:, "Exterior1st"].mode()[0]
df_main.loc[:, "Exterior2nd"] = df_main.loc[:, "Exterior2nd"].mode()[0]
df_main.loc[:, "TotalBsmtSF"] = df_main.loc[:, "TotalBsmtSF"].mean()
df_main.loc[:, "SaleType"] = df_main.loc[:, "SaleType"].fillna("Oth")
df_main.loc[:, "BsmtFinType1"] = df_main.loc[:, "BsmtFinType1"].mode()[0]
df_main.loc[:, "BsmtFinType2"] = df_main.loc[:, "BsmtFinType2"].mode()[0]
df_main.loc[:, "BsmtFinSF1"] = df_main.loc[:, "BsmtFinSF1"].mean()
df_main.loc[:, "BsmtFinSF2"] = df_main.loc[:, "BsmtFinSF2"].mean()
df_main.loc[:, "BsmtFullBath"] = df_main.loc[:, "BsmtFullBath"].fillna(0)
df_main.loc[:, "BsmtHalfBath"] = df_main.loc[:, "BsmtHalfBath"].fillna(0)
df_main.loc[:, "BsmtUnfSF"] = df_main.loc[:, "BsmtUnfSF"].fillna(0)
# CentralAir : NA most likely means No
df_main.loc[:, "CentralAir"] = df_main.loc[:, "CentralAir"].fillna("N")
# Condition : NA most likely means Normal
df_main.loc[:, "Condition1"] = df_main.loc[:, "Condition1"].fillna("Norm")
df_main.loc[:, "Condition2"] = df_main.loc[:, "Condition2"].fillna("Norm")
# EnclosedPorch : NA most likely means no enclosed porch
df_main.loc[:, "EnclosedPorch"] = df_main.loc[:, "EnclosedPorch"].fillna(0)
# External stuff : NA most likely means average
df_main.loc[:, "ExterCond"] = df_main.loc[:, "ExterCond"].fillna("TA")
df_main.loc[:, "ExterQual"] = df_main.loc[:, "ExterQual"].fillna("TA")
# Fence : data description says NA means "no fence"
df_main.loc[:, "Fence"] = df_main.loc[:, "Fence"].fillna("No")
# FireplaceQu : data description says NA means "no fireplace"
df_main.loc[:, "FireplaceQu"] = df_main.loc[:, "FireplaceQu"].fillna("No")
df_main.loc[:, "Fireplaces"] = df_main.loc[:, "Fireplaces"].fillna(0)
# Functional : data description says NA means typical
df_main.loc[:, "Functional"] = df_main.loc[:, "Functional"].fillna("Typ")
# GarageType etc : data description says NA for garage features is "no garage"
df_main.loc[:, "GarageType"] = df_main.loc[:, "GarageType"].fillna("No")
df_main.loc[:, "GarageFinish"] = df_main.loc[:, "GarageFinish"].fillna("No")
df_main.loc[:, "GarageQual"] = df_main.loc[:, "GarageQual"].fillna("No")
df_main.loc[:, "GarageCond"] = df_main.loc[:, "GarageCond"].fillna("No")
df_main.loc[:, "GarageArea"] = df_main.loc[:, "GarageArea"].fillna(0)
df_main.loc[:, "GarageCars"] = df_main.loc[:, "GarageCars"].fillna(0)
# HalfBath : NA most likely means no half baths above grade
df_main.loc[:, "HalfBath"] = df_main.loc[:, "HalfBath"].fillna(0)
# HeatingQC : NA most likely means typical
df_main.loc[:, "HeatingQC"] = df_main.loc[:, "HeatingQC"].fillna("TA")
# KitchenAbvGr : NA most likely means 0
df_main.loc[:, "KitchenAbvGr"] = df_main.loc[:, "KitchenAbvGr"].fillna(0)
# KitchenQual : NA most likely means typical
df_main.loc[:, "KitchenQual"] = df_main.loc[:, "KitchenQual"].fillna("TA")
# LotFrontage : NA most likely means no lot frontage
df_main.loc[:, "LotFrontage"] = df_main.loc[:, "LotFrontage"].fillna(0)
# LotShape : NA most likely means regular
df_main.loc[:, "LotShape"] = df_main.loc[:, "LotShape"].fillna("Reg")
# MasVnrType : NA most likely means no veneer
df_main.loc[:, "MasVnrType"] = df_main.loc[:, "MasVnrType"].fillna("None")
df_main.loc[:, "MasVnrArea"] = df_main.loc[:, "MasVnrArea"].fillna(0)
# MiscFeature : data description says NA means "no misc feature"
df_main.loc[:, "MiscFeature"] = df_main.loc[:, "MiscFeature"].fillna("No")
df_main.loc[:, "MiscVal"] = df_main.loc[:, "MiscVal"].fillna(0)
# OpenPorchSF : NA most likely means no open porch
df_main.loc[:, "OpenPorchSF"] = df_main.loc[:, "OpenPorchSF"].fillna(0)
# PavedDrive : NA most likely means not paved
df_main.loc[:, "PavedDrive"] = df_main.loc[:, "PavedDrive"].fillna("N")
# PoolQC : data description says NA means "no pool"
df_main.loc[:, "PoolQC"] = df_main.loc[:, "PoolQC"].fillna("No")
df_main.loc[:, "PoolArea"] = df_main.loc[:, "PoolArea"].fillna(0)
# SaleCondition : NA most likely means normal sale
df_main.loc[:, "SaleCondition"] = df_main.loc[:, "SaleCondition"].fillna("Normal")
# ScreenPorch : NA most likely means no screen porch
df_main.loc[:, "ScreenPorch"] = df_main.loc[:, "ScreenPorch"].fillna(0)
# TotRmsAbvGrd : NA most likely means 0
df_main.loc[:, "TotRmsAbvGrd"] = df_main.loc[:, "TotRmsAbvGrd"].fillna(0)
# Utilities : NA most likely means all public utilities
df_main.loc[:, "Utilities"] = df_main.loc[:, "Utilities"].fillna("AllPub")
# WoodDeckSF : NA most likely means no wood deck
df_main.loc[:, "WoodDeckSF"] = df_main.loc[:, "WoodDeckSF"].fillna(0)
df_main.loc[:, "Electrical"] = df_main.loc[:, "Electrical"].fillna("No")

var = 'GarageCars'
data = pd.concat([df_main['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
plt.show()



#Creating New Variables

for df in [df_main]:
    df["GarAreaPerCar"] = (df["GarageArea"] / df["GarageCars"]).fillna(0) #Bir araca düşen metrekare alanı
    df["GrLivAreaPerRoom"] = df["GrLivArea"] / df["TotRmsAbvGrd"] # Oda sayısı başına düşen brüt metrekare
    df["TotalHouseSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalFullBath"] = df["FullBath"] + df["BsmtFullBath"]
    df["TotalHalfBath"] = df["HalfBath"] + df["BsmtHalfBath"]
    df["InitHouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodHouseAge"] = df["InitHouseAge"] - (df["YrSold"] - df["YearRemodAdd"])
    df["IsRemod"] = (df["YearRemodAdd"] - df["YrSold"]).apply(lambda x: 1 if x > 0 else 0)
    df["IsGarage"] = df["GarageYrBlt"].apply(lambda x: 1 if x > 0 else 0)
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df["AvgQualCond"] = (df["OverallQual"] + df["OverallCond"]) / 2


for df in [df_main]:
    df.drop([
        "GarageArea", "GarageCars", "GrLivArea",
        "TotRmsAbvGrd", "TotalBsmtSF", "1stFlrSF",
        "2ndFlrSF", "FullBath", "BsmtFullBath", "HalfBath",
        "BsmtHalfBath", "YrSold", "YearBuilt", "YearRemodAdd",
        "GarageYrBlt", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
        "ScreenPorch", "OverallQual", "OverallCond"
    ], axis=1, inplace=True)

for col in df_main:
    if df_main[col].isnull().sum() != 0:
        print(f'{col} : {df_main[col].isnull().sum()} null values | Dtype : {df_main[col].dtype}')


##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df_main,cat_th=26, car_th=26)

cat_cols
num_cols
cat_but_car

##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def cat_summary_l(dataframe, cat_cols, plot=False):
    for col_name in cat_cols:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show()

cat_summary_l(df_main, cat_cols)

for col in cat_cols:
    cat_summary(df_main,col)


##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df_main, col, plot=False)



##################################
# NUMERİK DEĞİŞKENLERİN TARGET GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df_main, "SalePrice", col)



##################################
# AYKIRI DEĞER ANALİZİ
##################################

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df_main, col))
    if check_outlier(df_main, col):
        replace_with_thresholds(df_main, col)


##################################
# KORELASYON
##################################

# Korelasyon, olasılık kuramı ve istatistikte iki rassal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü belirtir

df_main.corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_main.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df_main, col)


# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["SalePrice"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df_main, cat_cols, drop_first=True)

df.shape

##################################
# MODELLEME
##################################
final_test_df = df[df['SalePrice'].isnull()]
final_train_df = df[~df['SalePrice'].isnull()]
X_train = final_train_df.drop(["SalePrice"], axis=1)
y_train = final_train_df.SalePrice

X_test = final_test_df.drop(["SalePrice"], axis=1)

X_train.shape
y_train.shape
X_test.shape

for col in X_train:
    if X_test[col].isnull().sum() == 0:
        print(f'{col} : {X_test[col].isnull().sum()} null values | Dtype : {X_test[col].dtype}')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_log = np.log10(y_train)

regressors = {
    "XGB Regressor": XGBRegressor(),
    "LGBM Regressor": LGBMRegressor(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "Elastic Net": ElasticNet(),
    "Bayesian Ridge": BayesianRidge(),
    "SVR": SVR(),
    "GB Regressor": GradientBoostingRegressor(random_state=0)
}

results = pd.DataFrame(columns=["Regressor", "Avg_RMSE"])
for name, reg in regressors.items():
    model = reg
    cv_results = cross_validate(
        model, X_train_scaled, y_train_log, cv=10,
        scoring=(['neg_root_mean_squared_error'])
    )

    results = results.append({
        "Regressor": name,
        "Avg_RMSE": np.abs(cv_results['test_neg_root_mean_squared_error']).mean()
    }, ignore_index=True)

results = results.sort_values("Avg_RMSE", ascending=True)
results.reset_index(drop=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=results, x="Avg_RMSE", y="Regressor")
plt.title("Average RMSE CV Score")
plt.show()


gbr = GradientBoostingRegressor(random_state=0)
params = {
    "loss": ("squared_error", "absolute_error"),
    "learning_rate": (1.0, 0.1, 0.01),
    "n_estimators": (50, 100, 200)
}
reg1 = GridSearchCV(gbr, params, cv=10)
reg1.fit(X_train_scaled, y_train_log)
print("Best hyperparameter:", reg1.best_params_)

y_pred = reg1.predict(X_train_scaled)
print(f"Train RMSE: {mean_squared_error(y_train_log, y_pred, squared=False)}")

xgb = XGBRegressor(random_state=0)
params = {
    "max_depth": (3, 6, 9),
    "learning_rate": (0.3, 0.1, 0.05),
    "n_estimators": (50, 100, 200)
}
reg2 = GridSearchCV(xgb, params, cv=10)
reg2.fit(X_train_scaled, y_train_log)
print("Best hyperparameter:", reg2.best_params_)

y_pred = reg2.predict(X_train_scaled)
print(f"Train RMSE: {mean_squared_error(y_train_log, y_pred, squared=False)}")

lgbm = LGBMRegressor(random_state=0)
params = {
    "num_leaves": (11, 31, 51),
    "learning_rate": (0.5, 0.1, 0.05),
    "n_estimators": (50, 100, 200)
}
reg3 = GridSearchCV(lgbm, params, cv=10)
reg3.fit(X_train_scaled, y_train_log)
print("Best hyperparameter:", reg3.best_params_)

y_pred = reg3.predict(X_train_scaled)
print(f"Train RMSE: {mean_squared_error(y_train_log, y_pred, squared=False)}")

def sm_predict(X):
    return (3 * reg1.predict(X) + 2 * reg2.predict(X) + 5 * reg3.predict(X)) / 10

y_pred_stack = sm_predict(X_train_scaled)
print(f"Train RMSE with Stacking: {mean_squared_error(y_train_log, y_pred_stack, squared=False)}")

y_pred = sm_predict(X_test_scaled)
y_pred_inv = 10 ** y_pred

submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred_inv})
submission.to_csv('submission.csv', index=False)