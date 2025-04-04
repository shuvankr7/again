import streamlit as st
import pandas as pd
import os
from sms_extractor import SMSExtractor
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="SMS Transaction Extractor",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .success-text {
        color: #4CAF50;
    }
    .warning-text {
        color: #FF9800;
    }
    .error-text {
        color: #F44336;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .upload-box {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the extractor
@st.cache_resource
def get_extractor(auto_install_model=False):
    return SMSExtractor(auto_install_model=auto_install_model)

# Main app
def main():
    st.markdown('<h1 class="main-header">SMS Transaction Extractor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sub-header">About</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        This app extracts transaction data from SMS messages using NLP and regex patterns.
        It can identify amounts, dates, times, transaction IDs, account numbers, bank names,
        UPI IDs, transaction types, and balances.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="sub-header">Instructions</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        1. Enter SMS messages manually or upload a file<br>
        2. Click "Extract Data" to process the messages<br>
        3. View and download the extracted data
        </p>
        """, unsafe_allow_html=True)
        
        # Add a toggle for auto-installing the spaCy model
        auto_install = st.checkbox("Auto-install spaCy model if missing", value=False)
        if auto_install:
            st.info("The app will attempt to install the spaCy model if it's not already installed.")
        
        st.markdown('<h2 class="sub-header">Sample Messages</h2>', unsafe_allow_html=True)
        sample_messages = [
            "An amount of INR 5.00 has been DEBITED to your account XXX670 on 12/03/2025. Total Avail.bal INR 3,127.41. - Canara Bank",
            "Dear customer, Your account XXXXXXXXXX1234 is debited with Rs.1234.00 on 27/Aug/2024 19:59:09. Ref.No:123456789012",
            "Dear UPI user A/C X7712 debited by 265.94 on date 17Mar25 trf to BBNOW Refno 544272365695. -SBI"
        ]
        for i, msg in enumerate(sample_messages):
            st.text_area(f"Sample {i+1}", msg, height=100, key=f"sample_{i}")
    
    # Main content
    extractor = get_extractor(auto_install_model=auto_install)
    
    # Input method selection
    input_method = st.radio("Select input method:", ["Manual Entry", "File Upload"])
    
    messages = []
    
    if input_method == "Manual Entry":
        st.markdown('<h2 class="sub-header">Enter SMS Messages</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Enter one SMS message per line. You can paste multiple messages.
        </p>
        """, unsafe_allow_html=True)
        
        manual_input = st.text_area("SMS Messages", height=300)
        if manual_input:
            messages = [msg.strip() for msg in manual_input.split('\n') if msg.strip()]
    
    else:  # File Upload
        st.markdown('<h2 class="sub-header">Upload SMS Messages</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p class="info-text">
        Upload a text file with one SMS message per line.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                file_content = uploaded_file.getvalue().decode("utf-8")
                messages = [msg.strip() for msg in file_content.split('\n') if msg.strip()]
                st.success(f"Successfully loaded {len(messages)} messages from file.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Extract button
    if st.button("Extract Data", key="extract_button"):
        if not messages:
            st.warning("Please enter or upload SMS messages first.")
        else:
            with st.spinner("Extracting transaction data..."):
                try:
                    # Process messages
                    df = extractor.process_messages(messages)
                    
                    if df.empty:
                        st.warning("No transaction data could be extracted from the provided messages.")
                    else:
                        st.success(f"Successfully extracted data from {len(df)} messages.")
                        
                        # Display results
                        st.markdown('<h2 class="sub-header">Extracted Data</h2>', unsafe_allow_html=True)
                        st.dataframe(df)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download as CSV",
                                data=csv,
                                file_name="transactions.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            json_str = df.to_json(orient="records", indent=4)
                            st.download_button(
                                label="Download as JSON",
                                data=json_str,
                                file_name="transactions.json",
                                mime="application/json"
                            )
                        
                        # Display statistics
                        st.markdown('<h2 class="sub-header">Statistics</h2>', unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Transactions", len(df))
                        
                        with col2:
                            credit_count = len(df[df['transaction_type'] == 'credit']) if 'transaction_type' in df.columns else 0
                            debit_count = len(df[df['transaction_type'] == 'debit']) if 'transaction_type' in df.columns else 0
                            st.metric("Credit/Debit", f"{credit_count}/{debit_count}")
                        
                        with col3:
                            total_amount = df['amount'].sum() if 'amount' in df.columns else 0
                            st.metric("Total Amount", f"₹{total_amount:.2f}")
                        
                        # Display bank distribution
                        if 'bank_name' in df.columns and not df['bank_name'].isna().all():
                            st.markdown('<h3 class="sub-header">Bank Distribution</h3>', unsafe_allow_html=True)
                            bank_counts = df['bank_name'].value_counts()
                            st.bar_chart(bank_counts)
                
                except Exception as e:
                    st.error(f"Error extracting data: {e}")
                    logger.error(f"Error in extraction: {e}", exc_info=True)

if __name__ == "__main__":
    main() 
