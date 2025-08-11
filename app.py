import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sydney Real Estate Analytics",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .price-prediction {
        font-size: 2rem;
        font-weight: bold;
        color: #2ca02c;
        text-align: center;
        padding: 1rem;
        background-color: #f0f8f0;
        border-radius: 0.5rem;
        border: 2px solid #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load all necessary data files"""
    try:
        # Load suburb info
        suburb_info = pd.read_csv('suburb_info.csv')
        
        # Load Sydney suburbs reviews data
        suburbs_reviews = pd.read_csv('Sydney-Suburbs-Reviews.csv')
        
        # Load properties data for analytics
        properties = pd.read_csv('domain_properties.csv')
        
        return suburb_info, suburbs_reviews, properties
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Suburb recommender class (simplified version)
class EthnicBudgetRecommender:
    """Recommender based on ethnic background preference and budget"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.ethnicities = self.get_unique_ethnicities()
        
    def parse_ethnic_breakdown(self, ethnic_str):
        """Parse ethnic breakdown string into dictionary with percentages"""
        if pd.isna(ethnic_str):
            return {}
        
        ethnic_dict = {}
        pattern = r'([A-Za-z\s]+?)\s+([\d.]+)%'
        matches = re.findall(pattern, ethnic_str)
        
        for ethnicity, percentage in matches:
            ethnicity = ethnicity.strip()
            ethnic_dict[ethnicity] = float(percentage)
        
        return ethnic_dict
    
    def get_unique_ethnicities(self):
        """Extract all unique ethnicities from the dataset"""
        all_ethnicities = set()
        for _, row in self.df.iterrows():
            if pd.notna(row['Ethnic Breakdown 2016']):
                ethnic_dict = self.parse_ethnic_breakdown(row['Ethnic Breakdown 2016'])
                all_ethnicities.update(ethnic_dict.keys())
        return sorted(list(all_ethnicities))
    
    def recommend_suburbs(self, target_ethnicity, max_weekly_rent, top_n=10):
        """Recommend suburbs based on ethnicity preference and budget"""
        
        # Clean rent data
        df_clean = self.df.copy()
        df_clean['Weekly_Rent'] = df_clean['Median House Rent (per week)'].str.replace('$', '').str.replace(',', '')
        df_clean['Weekly_Rent'] = pd.to_numeric(df_clean['Weekly_Rent'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Weekly_Rent'])
        
        # Filter by budget
        budget_filtered = df_clean[df_clean['Weekly_Rent'] <= max_weekly_rent].copy()
        
        if len(budget_filtered) == 0:
            return pd.DataFrame()
        
        # Calculate ethnicity percentage for target ethnicity
        budget_filtered['Ethnicity_Percentage'] = budget_filtered['Ethnic Breakdown 2016'].apply(
            lambda x: self.parse_ethnic_breakdown(x).get(target_ethnicity, 0)
        )
        
        # Calculate combined score
        budget_filtered['Rent_Score'] = 1 / (budget_filtered['Weekly_Rent'] / budget_filtered['Weekly_Rent'].min())
        budget_filtered['Combined_Score'] = (
            budget_filtered['Ethnicity_Percentage'] * 0.8 +
            budget_filtered['Rent_Score'] * 0.2
        )
        
        # Sort by combined score
        recommendations = budget_filtered.nlargest(top_n, 'Combined_Score')
        
        return recommendations[['Name', 'Weekly_Rent', 'Ethnicity_Percentage', 'Combined_Score', 'Region']].round(2)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Price Predictor'

# Sidebar navigation
st.sidebar.markdown('<div class="main-header">🏠 Sydney Real Estate</div>', unsafe_allow_html=True)

page = st.sidebar.selectbox(
    "Navigate to:",
    ["🏠 Price Predictor", "🏘️ Suburb Recommender", "📊 Analytics Dashboard"]
)

# Load data
suburb_info, suburbs_reviews, properties = load_data()
model = load_model()

if suburb_info is None or suburbs_reviews is None or properties is None or model is None:
    st.error("Failed to load necessary data files. Please ensure all CSV files and model.pkl are in the correct directory.")
    st.stop()

# ============================================================================
# PAGE 1: PRICE PREDICTOR
# ============================================================================

if page == "🏠 Price Predictor":
    st.markdown('<div class="main-header">🏠 Sydney House Price Predictor (2021 Edition)</div>', unsafe_allow_html=True)
    st.markdown("Predict house prices in Sydney based on property features and location (based on 2021 data).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Property Details</div>', unsafe_allow_html=True)
        
        num_bath = st.number_input("Number of Bathrooms", min_value=0, max_value=10, value=2)
        num_bed = st.number_input("Number of Bedrooms", min_value=0, max_value=10, value=3)
        num_parking = st.number_input("Number of Parking Spaces", min_value=0, max_value=10, value=1)
        property_size = st.number_input("Property Size (sqm)", min_value=50, max_value=5000, value=500)
        
    with col2:
        st.markdown('<div class="sub-header">Location & Market</div>', unsafe_allow_html=True)
        
        # Suburb selection
        suburb_options = suburb_info['suburb'].tolist()
        selected_suburb = st.selectbox("Select Suburb", suburb_options)
        
        # Get suburb data
        suburb_data = suburb_info[suburb_info['suburb'] == selected_suburb].iloc[0]
        suburb_population = suburb_data['suburb_population']
        suburb_median_income = suburb_data['suburb_median_income']
        
        # Display suburb info
        st.info(f"**{selected_suburb}** - Population: {suburb_population:,}, Median Income: ${suburb_median_income:,}")
        
        cash_rate = st.number_input("Cash Rate (%)", min_value=0.0, max_value=10.0, value=4.35, step=0.1)
        property_inflation_index = st.number_input("Property Inflation Index", min_value=100.0, max_value=300.0, value=150.0, step=1.0)
        km_from_cbd = st.number_input("Distance from CBD (km)", min_value=0.0, max_value=150.0, value=25.0, step=1.0)
    
    # Prediction
    st.markdown('<div class="sub-header">Price Prediction</div>', unsafe_allow_html=True)
    
    if st.button("Predict House Price", type="primary"):
        # Prepare features for prediction
        features = np.array([[
            num_bath, num_bed, num_parking, property_size,
            suburb_population, suburb_median_income, cash_rate,
            property_inflation_index, km_from_cbd
        ]])
        
        # Make prediction
        predicted_price = model.predict(features)[0]
        
        # Calculate range (±18%)
        lower_bound = predicted_price * 0.82
        upper_bound = predicted_price * 1.18
        
        # Display prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Price", f"${predicted_price:,.0f}")
        with col2:
            st.metric("Lower Range (-18%)", f"${lower_bound:,.0f}")
        with col3:
            st.metric("Upper Range (+18%)", f"${upper_bound:,.0f}")
        
        st.markdown(f'<div class="price-prediction">Estimated Price: ${predicted_price:,.0f} ± 18%</div>', unsafe_allow_html=True)
        
        # Display feature importance info
        st.info("💡 **Tip:** Property size, location (suburb median income), and distance from CBD are typically the most important factors in price prediction.")

# ============================================================================
# PAGE 2: SUBURB RECOMMENDER
# ============================================================================

elif page == "🏘️ Suburb Recommender":
    st.markdown('<div class="main-header">🏘️ Sydney Suburb Recommender</div>', unsafe_allow_html=True)
    st.markdown("Find suburbs that match your ethnic background preferences and budget.")
    
    # Initialize recommender
    recommender = EthnicBudgetRecommender(suburbs_reviews)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Your Preferences</div>', unsafe_allow_html=True)
        
        # Ethnicity selection
        ethnicity_options = recommender.ethnicities
        selected_ethnicity = st.selectbox("Preferred Ethnic Background", ethnicity_options)
        
        # Budget input
        max_rent = st.number_input("Maximum Weekly Rent ($)", min_value=200, max_value=3000, value=800, step=50)
        
        # Number of recommendations
        num_recommendations = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
        
    with col2:
        st.markdown('<div class="sub-header">Search Results</div>', unsafe_allow_html=True)
        
        if st.button("Find Suburbs", type="primary"):
            recommendations = recommender.recommend_suburbs(selected_ethnicity, max_rent, num_recommendations)
            
            if len(recommendations) > 0:
                st.success(f"Found {len(recommendations)} suburbs matching your criteria!")
                
                # Display recommendations
                for i, (_, row) in enumerate(recommendations.iterrows(), 1):
                    with st.expander(f"{i}. {row['Name']} - ${row['Weekly_Rent']}/week"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Weekly Rent", f"${row['Weekly_Rent']}")
                            st.metric(f"{selected_ethnicity} Population", f"{row['Ethnicity_Percentage']}%")
                        with col_b:
                            st.metric("Overall Score", f"{row['Combined_Score']:.2f}")
                            st.metric("Region", row['Region'])
                
                # Show summary chart
                st.markdown('<div class="sub-header">Recommendations Overview</div>', unsafe_allow_html=True)
                
                fig = px.scatter(
                    recommendations, 
                    x='Weekly_Rent', 
                    y='Ethnicity_Percentage',
                    size='Combined_Score',
                    hover_name='Name',
                    color='Region',
                    title=f"Suburbs by Rent vs {selected_ethnicity} Population %",
                    labels={'Weekly_Rent': 'Weekly Rent ($)', 'Ethnicity_Percentage': f'{selected_ethnicity} Population (%)'}
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("No suburbs found matching your criteria. Try increasing your budget or selecting a different ethnicity.")

# ============================================================================
# PAGE 3: ANALYTICS DASHBOARD
# ============================================================================

elif page == "📊 Analytics Dashboard":
    st.markdown('<div class="main-header">📊 Sydney Real Estate Analytics</div>', unsafe_allow_html=True)
    st.markdown("Comprehensive insights into Sydney's real estate market.")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Trends", "🏘️ Suburb Analysis", "🏠 Property Types", "💰 Investment Insights"])
    
    with tab1:
        st.markdown('<div class="sub-header">Price Distribution Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution histogram
            fig_hist = px.histogram(
                properties, 
                x='price', 
                nbins=50,
                title="House Price Distribution",
                labels={'price': 'Price ($)', 'count': 'Number of Properties'}
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col2:
            # Price by distance from CBD
            fig_scatter = px.scatter(
                properties.sample(1000) if len(properties) > 1000 else properties,
                x='km_from_cbd',
                y='price',
                color='type',
                title="Price vs Distance from CBD",
                labels={'km_from_cbd': 'Distance from CBD (km)', 'price': 'Price ($)'}
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Price", f"${properties['price'].mean():,.0f}")
        with col2:
            st.metric("Median Price", f"${properties['price'].median():,.0f}")
        with col3:
            st.metric("Most Expensive", f"${properties['price'].max():,.0f}")
        with col4:
            st.metric("Most Affordable", f"${properties['price'].min():,.0f}")
    
    with tab2:
        st.markdown('<div class="sub-header">Suburb-wise Analysis</div>', unsafe_allow_html=True)
        
        # Top 10 most expensive suburbs
        suburb_avg_prices = properties.groupby('suburb')['price'].agg(['mean', 'count']).reset_index()
        suburb_avg_prices = suburb_avg_prices[suburb_avg_prices['count'] >= 5]  # At least 5 properties
        top_expensive = suburb_avg_prices.nlargest(10, 'mean')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                top_expensive,
                x='mean',
                y='suburb',
                orientation='h',
                title="Top 10 Most Expensive Suburbs (Average Price)",
                labels={'mean': 'Average Price ($)', 'suburb': 'Suburb'}
            )
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Population vs median income from suburb_info
            fig_pop_income = px.scatter(
                suburb_info,
                x='suburb_population',
                y='suburb_median_income',
                hover_name='suburb',
                title="Suburb Population vs Median Income",
                labels={'suburb_population': 'Population', 'suburb_median_income': 'Median Income ($)'}
            )
            fig_pop_income.update_layout(height=500)
            st.plotly_chart(fig_pop_income, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">Property Types Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Property type distribution
            type_counts = properties['type'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Distribution of Property Types"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Average price by property type
            avg_price_by_type = properties.groupby('type')['price'].mean().sort_values(ascending=False)
            fig_bar_type = px.bar(
                x=avg_price_by_type.index,
                y=avg_price_by_type.values,
                title="Average Price by Property Type",
                labels={'x': 'Property Type', 'y': 'Average Price ($)'}
            )
            fig_bar_type.update_layout(height=400)
            st.plotly_chart(fig_bar_type, use_container_width=True)
        
        # Property features analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Bedrooms vs Price
            bedroom_price = properties.groupby('num_bed')['price'].mean().reset_index()
            fig_bed = px.line(
                bedroom_price,
                x='num_bed',
                y='price',
                title="Average Price by Number of Bedrooms",
                markers=True
            )
            st.plotly_chart(fig_bed, use_container_width=True)
        
        with col2:
            # Property size vs Price correlation
            fig_size = px.scatter(
                properties.sample(1000) if len(properties) > 1000 else properties,
                x='property_size',
                y='price',
                title="Property Size vs Price",
                labels={'property_size': 'Property Size (sqm)', 'price': 'Price ($)'}
            )
            st.plotly_chart(fig_size, use_container_width=True)
    
    with tab4:
        st.markdown('<div class="sub-header">Investment Insights</div>', unsafe_allow_html=True)
        
        # Calculate price per sqm for better comparison
        properties_clean = properties[properties['property_size'] > 0].copy()
        properties_clean['price_per_sqm'] = properties_clean['price'] / properties_clean['property_size']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price per sqm by distance from CBD
            fig_psqm_cbd = px.scatter(
                properties_clean.sample(1000) if len(properties_clean) > 1000 else properties_clean,
                x='km_from_cbd',
                y='price_per_sqm',
                title="Price per sqm vs Distance from CBD",
                labels={'km_from_cbd': 'Distance from CBD (km)', 'price_per_sqm': 'Price per sqm ($)'}
            )
            st.plotly_chart(fig_psqm_cbd, use_container_width=True)
        
        with col2:
            # Cash rate impact visualization
            rate_price = properties.groupby('cash_rate')['price'].mean().reset_index()
            fig_rate = px.line(
                rate_price,
                x='cash_rate',
                y='price',
                title="Average Price by Cash Rate",
                markers=True,
                labels={'cash_rate': 'Cash Rate (%)', 'price': 'Average Price ($)'}
            )
            st.plotly_chart(fig_rate, use_container_width=True)
        
        # Investment recommendations
        st.markdown('<div class="sub-header">💡 Investment Insights</div>', unsafe_allow_html=True)
        
        # Best value suburbs (high median income, lower prices)
        value_analysis = suburb_info.merge(
            properties.groupby('suburb')['price'].mean().reset_index(),
            on='suburb',
            how='inner'
        )
        value_analysis['value_score'] = value_analysis['suburb_median_income'] / (value_analysis['price'] / 1000)
        best_value = value_analysis.nlargest(5, 'value_score')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("🎯 **Best Value Suburbs** (High Income, Lower Prices)")
            for _, suburb in best_value.iterrows():
                st.write(f"**{suburb['suburb']}** - Avg Price: ${suburb['price']:,.0f}, Median Income: ${suburb['suburb_median_income']:,}")
        
        with col2:
            # Growth potential areas (far from CBD, lower prices)
            growth_potential = properties[
                (properties['km_from_cbd'] > 30) & 
                (properties['price'] < properties['price'].median())
            ]['suburb'].value_counts().head(5)
            
            st.info("🚀 **Growth Potential Areas** (Far from CBD, Affordable)")
            for suburb, count in growth_potential.items():
                avg_price = properties[properties['suburb'] == suburb]['price'].mean()
                st.write(f"**{suburb}** - {count} properties, Avg: ${avg_price:,.0f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Sydney Real Estate Analytics**")
st.sidebar.markdown("Built with Streamlit & XGBoost")
st.sidebar.markdown("Data includes 11K+ property sales")