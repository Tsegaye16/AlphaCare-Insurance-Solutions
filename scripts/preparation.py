import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


class EDA:
    def __init__(self, data):
        self.data = data

    def missing_data_summary(self) -> pd.DataFrame:
        """
        Returns a summary of columns with missing data, including count and percentage of missing values.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Missing Count' and 'Percentage (%)' for columns with missing values.
        """
        # Total missing values per column
        missing_data = self.data.isnull().sum()
        
        # Filter only columns with missing values greater than 0
        missing_data = missing_data[missing_data > 0]
        
        # Calculate the percentage of missing data
        missing_percentage = (missing_data / len(self.data)) * 100
        
        # Combine the counts and percentages into a DataFrame
        missing_df = pd.DataFrame({
            'Missing Count': missing_data, 
            'Percentage (%)': missing_percentage
        })
        
        # Sort by percentage of missing data
        missing_df = missing_df.sort_values(by='Percentage (%)', ascending=False)
        
        return missing_df
    
    def handle_missing_data(self, missing_type: str, missing_cols: list) -> pd.DataFrame:
        """
        Handles missing data based on predefined strategies.
        """
        if missing_type == 'high':
            # Drop columns with high missing data
            self.data = self.data.drop(columns=missing_cols, errors='ignore')
        elif missing_type == 'moderate':
            # Impute or drop columns with moderate missing data
            for col in missing_cols:
                if col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        # Impute categorical columns with mode (check if mode exists)
                        if not self.data[col].mode().empty:
                            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                        else:
                            self.data[col] = self.data[col].fillna('Unknown')  # Default for empty mode
                    else:
                        # Impute numerical columns with median (check if median exists)
                        if not self.data[col].isnull().all():  # Ensure column has some numeric values
                            self.data[col] = self.data[col].fillna(self.data[col].median())
                        else:
                            self.data[col] = self.data[col].fillna(0)  # Default for empty median
        else:
            # Handle low missing data (default)
            for col in missing_cols:
                if col in self.data.columns:
                    if self.data[col].dtype == 'object':
                        if not self.data[col].mode().empty:
                            self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
                        else:
                            self.data[col] = self.data[col].fillna('Unknown')  # Default for empty mode
                    else:
                        if not self.data[col].isnull().all():
                            self.data[col] = self.data[col].fillna(self.data[col].median())
                        else:
                            self.data[col] = self.data[col].fillna(0)  # Default for empty median

        return self.data

    def boolean_to_integer(self, column_name:str):
        self.data[column_name] = self.data[column_name].astype(int)
        return self.data
    def categorical_summary(self) -> pd.DataFrame:
        """
        Returns a summary of categorical columns, including column name, data type, and number of unique values.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Column', 'DataType', and 'NumUniqueValues'.
        """
        # Get categorical columns
        categorical_columns = self.data.select_dtypes(include='object').columns

        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Column': categorical_columns,
            'DataType': [self.data[col].dtype for col in categorical_columns],
            'NumUniqueValues': [self.data[col].nunique() for col in categorical_columns]
        })

        return summary_df

    def add_transaction_date_features(self) -> pd.DataFrame:
        """
        Extracts year, month, and quarter features from a 'TransactionMonth' datetime column.

        Returns:
            pd.DataFrame: The updated DataFrame with new date-related columns.
        """
        self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'])
        self.data['TransactionYear'] = self.data['TransactionMonth'].dt.year
        self.data['TransactionMonthOnly'] = self.data['TransactionMonth'].dt.month
        self.data['TransactionQuarter'] = self.data['TransactionMonth'].dt.quarter
        # Drop the original datetime column if it's no longer needed
        self.data = self.data.drop(columns=['TransactionMonth'])
        return self.data


    def map_title_to_gender(self, title_column: str, gender_column: str) -> pd.DataFrame:
        """
        Maps titles to gender and fills missing gender values based on the title.

        Args:
            title_column (str): The column name containing titles (e.g., 'Title').
            gender_column (str): The column name containing gender (e.g., 'Gender').

        Returns:
            pd.DataFrame: The updated DataFrame with filled gender values.
        """
        title_to_gender_map = {
            'Mr': 'Male',
            'Mrs': 'Female',
            'Ms': 'Female',
            'Miss': 'Female',
            'Dr': None  # Leave as None since it could be either gender
        }

        self.data[gender_column] = self.data.apply(
            lambda row: title_to_gender_map.get(row[title_column], row[gender_column]), axis=1
        )

        return self.data

    def encode_categorical_features(self, low_cardinality_threshold=10, high_cardinality_threshold=20) -> pd.DataFrame:
        """
        Encodes categorical features based on their cardinality. Low cardinality features are label encoded,
        moderate cardinality features are one-hot encoded, and high cardinality features are frequency encoded.

        Args:
            low_cardinality_threshold (int): The threshold for low cardinality features.
            high_cardinality_threshold (int): The threshold for high cardinality features.

        Returns:
            pd.DataFrame: The updated DataFrame with encoded features.
        """
        categorical_cols = self.data.select_dtypes(include='object').columns

        low_cardinality_features = [col for col in categorical_cols if self.data[col].nunique() <= low_cardinality_threshold]
        moderate_cardinality_features = [col for col in categorical_cols if low_cardinality_threshold < self.data[col].nunique() <= high_cardinality_threshold]
        high_cardinality_features = [col for col in categorical_cols if self.data[col].nunique() > high_cardinality_threshold]

        # Label Encoding for low cardinality features
        label_encoders = {}
        for feature in low_cardinality_features:
            le = LabelEncoder()
            self.data[feature] = le.fit_transform(self.data[feature])
            label_encoders[feature] = le

        # One-Hot Encoding for moderate cardinality features
        one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        for feature in moderate_cardinality_features:
            encoded = one_hot_encoder.fit_transform(self.data[[feature]])
            encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out([feature]))
            self.data = pd.concat([self.data, encoded_df], axis=1)
            self.data = self.data.drop(columns=[feature])

        # Frequency Encoding for high cardinality features
        for feature in high_cardinality_features:
            freq = self.data[feature].value_counts()
            self.data[feature + '_freq'] = self.data[feature].map(freq)
            self.data = self.data.drop(columns=[feature])

        return self.data

    def scale_and_select_features(self, target_column: str, k: int = 20) -> pd.DataFrame:
        """
        Scales the features, performs feature selection, and returns a new dataframe with selected features.

        Args:
            target_column (str): The target column for prediction.
            k (int): Number of top features to select using SelectKBest.

        Returns:
            pd.DataFrame: A DataFrame with scaled and selected features.
        """
        # Separate features and target
        X = self.data.drop(columns=target_column)
        y = self.data[target_column]

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Feature selection using SelectKBest
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X_scaled, y)

        # Get the selected features
        selected_features = X.columns[selector.get_support()]
        df_new = pd.DataFrame(X_new, columns=selected_features)
        df_new[target_column] = y

        return df_new

    def apply_pca(self, target_column: str, n_components: int = 20) -> pd.DataFrame:
        """
        Applies PCA to the features and returns the transformed dataset with the target variable.

        Args:
            target_column (str): The target column for prediction.
            n_components (int): Number of principal components.

        Returns:
            pd.DataFrame: A DataFrame with the principal components and the target variable.
        """
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Create a DataFrame with PCA components and the target variable
        pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns)
        df_pca[target_column] = y

        return df_pca

    def split_data(self, target_column: str, test_size: float = 0.2):
        """
        Splits the data into training and testing sets.

        Args:
            target_column (str): The target column for prediction.
            test_size (float): Proportion of the data to include in the test split.

        Returns:
            X_train, X_test, y_train, y_test: Training and testing features and target variables.
        """
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        return X_train, X_test, y_train, y_test