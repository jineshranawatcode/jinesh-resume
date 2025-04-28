#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Required packages installation:
# pip install pandas numpy matplotlib seaborn scikit-learn plotly

"""
Data Engineering Demo
--------------------
This script demonstrates various data engineering capabilities using Python 3.11.
It includes data generation, transformation, analysis, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import csv
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataEngineeringDemo:
    """
    A comprehensive class demonstrating various data engineering capabilities.
    This class includes methods for data generation, transformation, analysis,
    and visualization using sample data.
    """
    
    def __init__(self, seed: int = 42) -> None:
        """
        Initialize the DataEngineeringDemo class.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        self.data: Optional[pd.DataFrame] = None
        self.transformed_data: Optional[pd.DataFrame] = None
        self.logger = logger
        
    def generate_sample_data(self, num_rows: int = 1000) -> pd.DataFrame:
        """
        Generate sample data for demonstration purposes.
        
        Args:
            num_rows: Number of rows to generate
            
        Returns:
            DataFrame containing sample data
        """
        self.logger.info(f"Generating {num_rows} rows of sample data")
        
        # Generate dates
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(num_rows)]
        
        # Generate customer IDs
        customer_ids = [f"CUST_{i:04d}" for i in range(1, num_rows + 1)]
        
        # Generate product categories
        categories: np.ndarray = np.random.choice(
            ["Electronics", "Clothing", "Food", "Books", "Home", "Sports"],
            num_rows,
            p=[0.3, 0.2, 0.15, 0.1, 0.15, 0.1]
        )
        
        # Generate product prices
        prices: np.ndarray = np.random.uniform(10, 1000, num_rows)
        
        # Generate quantities
        quantities: np.ndarray = np.random.randint(1, 10, num_rows)
        
        # Calculate total amount
        total_amount: np.ndarray = prices * quantities
        
        # Generate customer satisfaction scores
        satisfaction: np.ndarray = np.random.normal(4.2, 0.8, num_rows)
        satisfaction = np.clip(satisfaction, 1, 5)  # Clip to range [1, 5]
        
        # Generate shipping times (in days)
        shipping_times: np.ndarray = np.random.exponential(3, num_rows)
        shipping_times = np.round(shipping_times).astype(int)
        
        # Generate regions
        regions: np.ndarray = np.random.choice(
            ["North", "South", "East", "West", "Central"],
            num_rows,
            p=[0.25, 0.25, 0.2, 0.2, 0.1]
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'customer_id': customer_ids,
            'product_category': categories,
            'price': prices,
            'quantity': quantities,
            'total_amount': total_amount,
            'satisfaction': satisfaction,
            'shipping_time': shipping_times,
            'region': regions
        })
        
        # Add some missing values
        missing_indices: np.ndarray = np.random.choice(num_rows, size=int(num_rows * 0.05), replace=False)
        df.loc[missing_indices, 'satisfaction'] = np.nan
        
        # Add some outliers
        outlier_indices: np.ndarray = np.random.choice(num_rows, size=int(num_rows * 0.02), replace=False)
        df.loc[outlier_indices, 'total_amount'] *= 10
        
        self.data = df
        self.logger.info(f"Generated sample data with shape: {df.shape}")
        return df
    
    def data_quality_check(self) -> Dict[str, Any]:
        """
        Perform data quality checks on the dataset.
        
        Returns:
            Dictionary containing data quality metrics
        """
        if self.data is None:
            self.logger.error("No data available. Please generate sample data first.")
            return {}
        
        self.logger.info("Performing data quality checks")
        
        quality_metrics = {
            'total_rows': len(self.data),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'duplicate_percentage': (self.data.duplicated().sum() / len(self.data) * 100),
            'data_types': self.data.dtypes.astype(str).to_dict(),
            'numeric_columns': self.data.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': self.data.select_dtypes(include=['object', 'category']).columns.tolist(),
            'date_columns': self.data.select_dtypes(include=['datetime']).columns.tolist()
        }
        
        # Check for outliers in numeric columns
        outliers = {}
        for col in quality_metrics['numeric_columns']:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)][col].count()
        
        quality_metrics['outliers'] = outliers
        
        self.logger.info(f"Data quality check completed. Found {quality_metrics['duplicate_rows']} duplicate rows.")
        return quality_metrics
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and outliers.
        
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            self.logger.error("No data available. Please generate sample data first.")
            return pd.DataFrame()
        
        self.logger.info("Cleaning data")
        
        # Make a copy to avoid modifying the original data
        df = self.data.copy()
        
        # Handle missing values
        # For numeric columns, fill with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill with mode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Handle outliers using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        self.transformed_data = df
        self.logger.info(f"Data cleaning completed. Shape after cleaning: {df.shape}")
        return df
    
    def transform_data(self) -> pd.DataFrame:
        """
        Transform the data by adding derived features and normalizing numeric columns.
        
        Returns:
            Transformed DataFrame
        """
        if self.data is None:
            self.logger.error("No data available. Please generate sample data first.")
            return pd.DataFrame()
        
        self.logger.info("Transforming data")
        
        # Use cleaned data if available, otherwise use original data
        df = self.transformed_data.copy() if self.transformed_data is not None else self.data.copy()
        
        # Extract date components
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        # Create derived features
        df['price_per_unit'] = df['total_amount'] / df['quantity']
        df['is_high_value'] = df['total_amount'] > df['total_amount'].median()
        df['satisfaction_level'] = pd.cut(
            df['satisfaction'],
            bins=[0, 2, 3, 4, 5],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        
        # Normalize numeric columns
        numeric_cols = ['price', 'quantity', 'total_amount', 'shipping_time']
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # One-hot encode categorical columns
        categorical_cols = ['product_category', 'region']
        df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        
        self.transformed_data = df
        self.logger.info(f"Data transformation completed. Shape after transformation: {df.shape}")
        return df
    
    def perform_pca(self, n_components: int = 2) -> Tuple[pd.DataFrame, PCA]:
        """
        Perform Principal Component Analysis on numeric columns.
        
        Args:
            n_components: Number of components to keep
            
        Returns:
            Tuple containing DataFrame with PCA results and PCA model
        """
        if self.transformed_data is None:
            self.logger.error("No transformed data available. Please transform data first.")
            return pd.DataFrame(), None
        
        self.logger.info(f"Performing PCA with {n_components} components")
        
        # Select numeric columns
        numeric_cols = self.transformed_data.select_dtypes(include=['number']).columns
        X = self.transformed_data[numeric_cols]
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add original index
        pca_df.index = self.transformed_data.index
        
        # Calculate explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        self.logger.info(f"PCA completed. Explained variance ratio: {explained_variance}")
        self.logger.info(f"Cumulative explained variance: {cumulative_variance}")
        
        return pca_df, pca
    
    def perform_clustering(self, n_clusters: int = 3) -> Tuple[pd.DataFrame, KMeans]:
        """
        Perform K-means clustering on numeric columns.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            Tuple containing DataFrame with cluster labels and KMeans model
        """
        if self.transformed_data is None:
            self.logger.error("No transformed data available. Please transform data first.")
            return pd.DataFrame(), None
        
        self.logger.info(f"Performing K-means clustering with {n_clusters} clusters")
        
        # Select numeric columns
        numeric_cols = self.transformed_data.select_dtypes(include=['number']).columns
        X = self.transformed_data[numeric_cols]
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        cluster_labels = kmeans.fit_predict(X)
        
        # Create DataFrame with cluster labels
        cluster_df = pd.DataFrame({'cluster': cluster_labels})
        cluster_df.index = self.transformed_data.index
        
        # Calculate cluster sizes
        cluster_sizes = cluster_df['cluster'].value_counts().sort_index()
        
        self.logger.info(f"Clustering completed. Cluster sizes: {cluster_sizes.to_dict()}")
        
        return cluster_df, kmeans
    
    def analyze_data(self) -> Dict:
        """
        Perform comprehensive data analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        if self.data is None:
            self.logger.error("No data available. Please generate sample data first.")
            return {}
        
        self.logger.info("Performing comprehensive data analysis")
        
        analysis_results = {}
        
        # Basic statistics
        analysis_results['basic_stats'] = self.data.describe().to_dict()
        
        # Correlation analysis
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        analysis_results['correlation'] = self.data[numeric_cols].corr().to_dict()
        
        # Time series analysis
        if 'date' in self.data.columns:
            daily_sales = self.data.groupby(self.data['date'].dt.date)['total_amount'].sum().reset_index()
            analysis_results['daily_sales'] = daily_sales.to_dict('records')
            
            monthly_sales = self.data.groupby(self.data['date'].dt.to_period('M'))['total_amount'].sum().reset_index()
            monthly_sales['date'] = monthly_sales['date'].astype(str)
            analysis_results['monthly_sales'] = monthly_sales.to_dict('records')
        
        # Category analysis
        if 'product_category' in self.data.columns:
            category_stats = self.data.groupby('product_category').agg({
                'total_amount': ['sum', 'mean', 'count'],
                'satisfaction': 'mean'
            }).reset_index()
            category_stats.columns = ['product_category', 'total_sales', 'avg_sale', 'transaction_count', 'avg_satisfaction']
            analysis_results['category_stats'] = category_stats.to_dict('records')
        
        # Regional analysis
        if 'region' in self.data.columns:
            region_stats = self.data.groupby('region').agg({
                'total_amount': ['sum', 'mean'],
                'satisfaction': 'mean'
            }).reset_index()
            region_stats.columns = ['region', 'total_sales', 'avg_sale', 'avg_satisfaction']
            analysis_results['region_stats'] = region_stats.to_dict('records')
        
        # Customer analysis
        if 'customer_id' in self.data.columns:
            customer_stats = self.data.groupby('customer_id').agg({
                'total_amount': ['sum', 'count'],
                'satisfaction': 'mean'
            }).reset_index()
            customer_stats.columns = ['customer_id', 'total_spent', 'transaction_count', 'avg_satisfaction']
            analysis_results['customer_stats'] = customer_stats.to_dict('records')
        
        self.logger.info("Data analysis completed")
        return analysis_results
    
    def visualize_data(self, save_path: Optional[str] = None) -> None:
        """
        Create visualizations for the data.
        
        Args:
            save_path: Directory to save visualizations (optional)
        """
        if self.data is None:
            self.logger.error("No data available. Please generate sample data first.")
            return
        
        self.logger.info("Creating visualizations")
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figures directory if save_path is provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        
        # 1. Time series of daily sales
        plt.figure(figsize=(12, 6))
        daily_sales = self.data.groupby(self.data['date'].dt.date)['total_amount'].sum().reset_index()
        plt.plot(daily_sales['date'], daily_sales['total_amount'], marker='o')
        plt.title('Daily Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Sales Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'daily_sales.png'))
        plt.close()
        
        # 2. Sales by product category
        plt.figure(figsize=(10, 6))
        category_sales = self.data.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
        sns.barplot(x=category_sales.index, y=category_sales.values)
        plt.title('Total Sales by Product Category')
        plt.xlabel('Product Category')
        plt.ylabel('Total Sales Amount')
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'category_sales.png'))
        plt.close()
        
        # 3. Customer satisfaction distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['satisfaction'], bins=20, kde=True)
        plt.title('Distribution of Customer Satisfaction')
        plt.xlabel('Satisfaction Score')
        plt.ylabel('Frequency')
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'satisfaction_distribution.png'))
        plt.close()
        
        # 4. Correlation heatmap
        plt.figure(figsize=(10, 8))
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        correlation = self.data[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'correlation_heatmap.png'))
        plt.close()
        
        # 5. Regional sales comparison
        plt.figure(figsize=(10, 6))
        region_sales = self.data.groupby('region')['total_amount'].sum()
        plt.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', startangle=90)
        plt.title('Sales Distribution by Region')
        plt.axis('equal')
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'regional_sales.png'))
        plt.close()
        
        # 6. Shipping time vs satisfaction
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='shipping_time', y='satisfaction', data=self.data, alpha=0.6)
        plt.title('Shipping Time vs Customer Satisfaction')
        plt.xlabel('Shipping Time (days)')
        plt.ylabel('Satisfaction Score')
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'shipping_vs_satisfaction.png'))
        plt.close()
        
        # 7. Interactive visualization using Plotly
        fig = px.scatter(
            self.data,
            x='total_amount',
            y='satisfaction',
            color='product_category',
            size='quantity',
            hover_data=['customer_id', 'region'],
            title='Relationship Between Amount, Satisfaction, and Product Category'
        )
        if save_path:
            fig.write_html(os.path.join(save_path, 'interactive_scatter.html'))
        
        self.logger.info("Visualizations created successfully")
    
    def export_data(self, format: str = 'csv', path: Optional[str] = None) -> str:
        """
        Export the data to a file.
        
        Args:
            format: Export format ('csv', 'json', 'excel')
            path: Path to save the file (optional)
            
        Returns:
            Path to the exported file
        """
        if self.data is None:
            self.logger.error("No data available. Please generate sample data first.")
            return ""
        
        self.logger.info(f"Exporting data to {format} format")
        
        # Use transformed data if available, otherwise use original data
        df = self.transformed_data if self.transformed_data is not None else self.data
        
        # Generate default path if not provided
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"data_export_{timestamp}.{format}"
        
        # Export based on format
        if format.lower() == 'csv':
            df.to_csv(path, index=False)
        elif format.lower() == 'json':
            df.to_json(path, orient='records', date_format='iso')
        elif format.lower() == 'excel':
            df.to_excel(path, index=False)
        else:
            self.logger.error(f"Unsupported export format: {format}")
            return ""
        
        self.logger.info(f"Data exported successfully to {path}")
        return path
    
    def create_data_pipeline(self) -> None:
        """
        Demonstrate a complete data pipeline from generation to export.
        """
        self.logger.info("Starting complete data pipeline demonstration")
        
        # 1. Generate sample data
        self.generate_sample_data(1000)
        
        # 2. Perform data quality check
        quality_metrics = self.data_quality_check()
        self.logger.info(f"Data quality metrics: {quality_metrics}")
        
        # 3. Clean data
        self.clean_data()
        
        # 4. Transform data
        self.transform_data()
        
        # 5. Perform PCA
        pca_df, pca = self.perform_pca(2)
        self.logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        # 6. Perform clustering
        cluster_df, kmeans = self.perform_clustering(3)
        self.logger.info(f"Cluster centers: {kmeans.cluster_centers_}")
        
        # 7. Analyze data
        analysis_results = self.analyze_data()
        self.logger.info("Data analysis completed")
        
        # 8. Visualize data
        self.visualize_data()
        
        # 9. Export data
        export_path = self.export_data('csv')
        self.logger.info(f"Data exported to {export_path}")
        
        self.logger.info("Data pipeline demonstration completed successfully")


def main():
    """
    Main function to demonstrate the DataEngineeringDemo class.
    """
    # Create an instance of DataEngineeringDemo
    demo = DataEngineeringDemo(seed=42)
    
    # Run the complete data pipeline
    demo.create_data_pipeline()
    
    # Alternatively, you can run individual steps:
    # demo.generate_sample_data(1000)
    # demo.data_quality_check()
    # demo.clean_data()
    # demo.transform_data()
    # demo.perform_pca(2)
    # demo.perform_clustering(3)
    # demo.analyze_data()
    # demo.visualize_data(save_path='visualizations')
    # demo.export_data('csv', 'data_export.csv')


if __name__ == "__main__":
    main()
