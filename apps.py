import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Load your trained XGBoost model
xgb_model = XGBClassifier()  # Load your model here

# Load the dataset
df = pd.read_csv(r"C:\Users\binu_\Music\robin_friend\Phishing_Legitimate_full.csv")

# Define a subset of relevant columns for interaction
#selected_columns = ['NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash', 'NumQueryComponents']
#[id	NumDots	SubdomainLevel	PathLevel	UrlLength	NumDash	NumDashInHostname	AtSymbol	TildeSymbol	NumUnderscore	NumPercent	NumQueryComponents	NumAmpersand	NumHash	NumNumericChars	NoHttps	RandomString	IpAddress	DomainInSubdomains	DomainInPaths	HttpsInHostname	HostnameLength	PathLength	QueryLength	DoubleSlashInPath	NumSensitiveWords	EmbeddedBrandName	PctExtHyperlinks	PctExtResourceUrls	ExtFavicon	InsecureForms	RelativeFormAction	ExtFormAction	AbnormalFormAction	PctNullSelfRedirectHyperlinks	FrequentDomainNameMismatch	FakeLinkInStatusBar	RightClickDisabled	PopUpWindow	SubmitInfoToEmail	IframeOrFrame	MissingTitle	ImagesOnlyInForm	SubdomainLevelRT	UrlLengthRT	PctExtResourceUrlsRT	AbnormalExtFormActionR	ExtMetaScriptLinkRT	PctExtNullSelfRedirectHyperlinksRT	CLASS_LABEL
#]
selected_columns=['NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname','AtSymbol','TildeSymbol','NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash',	'NumNumericChars','NoHttps','RandomString','IpAddress','DomainInSubdomains','DomainInPaths','HttpsInHostname','HostnameLength','PathLength','QueryLength','DoubleSlashInPath',	'NumSensitiveWords','EmbeddedBrandName','PctExtHyperlinks','PctExtResourceUrls','ExtFavicon','InsecureForms','RelativeFormAction','ExtFormAction','AbnormalFormAction',	'PctNullSelfRedirectHyperlinks','FrequentDomainNameMismatch','FakeLinkInStatusBar','RightClickDisabled','PopUpWindow','SubmitInfoToEmail','IframeOrFrame',	'MissingTitle','ImagesOnlyInForm','SubdomainLevelRT','UrlLengthRT','PctExtResourceUrlsRT','AbnormalExtFormActionR','ExtMetaScriptLinkRT','PctExtNullSelfRedirectHyperlinksRT']
# Extract features and labels
X = df[selected_columns]
y = df["CLASS_LABEL"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Fit the model to the training data
xgb_model.fit(X_train, y_train)

# Streamlit UI
st.title('Phishing URL Predictor')

# Create input fields for selected features
input_data = {}
for feature in selected_columns:
    input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)

if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    prediction = xgb_model.predict(input_df)
    result = "Legitimate" if prediction[0] == 1 else "Phished"
    st.write(f'The URL is predicted to be: {result}')
