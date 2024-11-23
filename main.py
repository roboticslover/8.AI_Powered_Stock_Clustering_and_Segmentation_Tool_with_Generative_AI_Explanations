# Import Necessary Libraries and Set Up Environment
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.preprocessing import StandardScaler  # For data scaling
from sklearn.cluster import KMeans  # Clustering algorithm
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced data visualization
import yfinance as yf  # For fetching stock data
import openai  # For interfacing with GPT
import streamlit as st  # For building the web app
import pickle  # For file handling
import os  # For operating system interactions
from dotenv import load_dotenv  # For loading environment variables

# Load environment variables from a .env file (if needed)
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure your API key is set as an environment variable

# Function to fetch stock data
def fetch_stock_data(tickers, period='1y'):
    """
    Fetches historical stock data for given tickers.

    Parameters:
    tickers (list): List of stock ticker symbols.
    period (str): Time period for which to fetch data.

    Returns:
    DataFrame: A DataFrame containing adjusted close prices for the tickers.
    """
    try:
        data = yf.download(tickers, period=period)['Adj Close']
        if data.empty:
            raise ValueError("No data fetched. Please check the ticker symbols and network connection.")
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Function to extract features
def extract_features(data):
    """
    Extracts statistical features from stock return data.

    Parameters:
    data (DataFrame): DataFrame containing stock returns.

    Returns:
    DataFrame: A DataFrame with extracted features for each stock.
    """
    features = pd.DataFrame()
    features['Mean Return'] = data.mean()
    features['Volatility'] = data.std()
    features['Skewness'] = data.skew()
    features['Kurtosis'] = data.kurtosis()
    return features

# Function to determine optimal number of clusters
def determine_optimal_clusters(data):
    """
    Uses the Elbow Method to determine the optimal number of clusters.

    Parameters:
    data (DataFrame): Feature data for clustering.

    Returns:
    int: Optimal number of clusters.
    """
    inertia = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
    # Plot the Elbow Curve
    plt.figure(figsize=(8, 4))
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Number of Clusters k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    st.write("### Elbow Method For Optimal k")
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure after displaying
    # For simplicity, we'll choose 3 clusters
    return 3

# Function to generate explanations using GPT-3.5
def generate_cluster_explanation(cluster_id, data):
    """
    Generates an explanation for a given cluster using GPT-3.5.

    Parameters:
    cluster_id (int): The cluster number.
    data (DataFrame): Feature data with cluster assignments.

    Returns:
    str: Generated explanation text.
    """
    cluster_data = data[data['Cluster'] == cluster_id]
    description = cluster_data.describe().to_string()
    prompt = f"Provide a detailed explanation of the following stock cluster characteristics:\n{description}\nWhat does this imply for investors?"

    try:
        response = openai.Completion.create(
            engine='gpt-3.5-turbo',
            prompt=prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7,
        )
        explanation = response.choices[0].text.strip()
        return explanation
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
        return "An error occurred while generating the explanation."

# Function to save objects to file
def save_object(obj, filename):
    """
    Saves a Python object to a file using pickle.

    Parameters:
    obj: The object to save.
    filename (str): The filename for the saved object.

    Returns:
    None
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

# Function to load objects from file
def load_object(filename):
    """
    Loads a Python object from a pickle file.

    Parameters:
    filename (str): The filename of the saved object.

    Returns:
    The loaded object.
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
    else:
        st.error(f"File {filename} does not exist.")
        return None

# Cached function to generate pair plot
@st.cache_data
def generate_pairplot(data):
    """
    Generates a pair plot for the data.

    Parameters:
    data (DataFrame): Feature data with cluster assignments.

    Returns:
    Figure: The pair plot figure.
    """
    pairplot_fig = sns.pairplot(data, hue='Cluster', diag_kind='kde', palette='Set2')
    return pairplot_fig

# Function to run the Streamlit app
def run_app(data, explanations):
    """
    Runs the Streamlit app.

    Parameters:
    data (DataFrame): Feature data with cluster assignments.
    explanations (dict): Dictionary of cluster explanations.

    Returns:
    None
    """
    st.title('AI-Powered Stock Clustering and Segmentation Tool')
    st.write('This application clusters stocks based on financial features and provides AI-generated explanations for each cluster.')

    # Display cluster selection
    cluster_ids = data['Cluster'].unique()
    selected_cluster = st.selectbox('Select a Cluster', cluster_ids)

    # Display cluster data
    cluster_data = data[data['Cluster'] == selected_cluster]
    st.write(f'Stocks in Cluster {selected_cluster}:')
    st.table(cluster_data)

    # Display explanation
    st.write(f"**AI-Generated Explanation for Cluster {selected_cluster}:**")
    st.write(explanations.get(selected_cluster, "No explanation available."))

    # Display Visualization
    st.write('### Pair Plot of Features')
    pairplot_fig = generate_pairplot(data)
    st.pyplot(pairplot_fig.fig)

# Main function
def main():
    # List of stock tickers
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'JPM', 'V', 'JNJ', 'WMT']

    # Fetch data
    stock_data = fetch_stock_data(tickers)

    if stock_data is None:
        return

    # Preprocess the Data
    # Handle missing values by forward filling
    stock_data.fillna(method='ffill', inplace=True)

    # Calculate daily returns
    returns = stock_data.pct_change().dropna()

    # Standardize the returns
    scaler = StandardScaler()
    scaled_returns = scaler.fit_transform(returns)

    # Convert scaled data back to DataFrame for ease of use
    scaled_returns_df = pd.DataFrame(scaled_returns, index=returns.index, columns=returns.columns)

    # Extract features
    features_df = extract_features(scaled_returns_df)

    # Determine optimal clusters
    optimal_clusters = determine_optimal_clusters(features_df)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    features_df['Cluster'] = kmeans.fit_predict(features_df)

    # Generate explanations for each cluster
    cluster_explanations = {}
    for cluster_id in features_df['Cluster'].unique():
        explanation = generate_cluster_explanation(cluster_id, features_df)
        cluster_explanations[cluster_id] = explanation

    # Save the features DataFrame and explanations
    save_object(features_df, 'features_df.pkl')
    save_object(cluster_explanations, 'cluster_explanations.pkl')

    # Run the app
    run_app(features_df, cluster_explanations)

# Entry point
if __name__ == '__main__':
    try:
        # Check if the features DataFrame and explanations are already saved
        if os.path.exists('features_df.pkl') and os.path.exists('cluster_explanations.pkl'):
            features_df = load_object('features_df.pkl')
            cluster_explanations = load_object('cluster_explanations.pkl')
            if features_df is not None and cluster_explanations is not None:
                run_app(features_df, cluster_explanations)
            else:
                st.error("Failed to load saved data. Running main processing.")
                main()
        else:
            main()
    except Exception as e:
        st.error(f"An error occurred: {e}")
