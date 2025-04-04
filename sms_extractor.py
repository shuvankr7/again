import pandas as pd
import re
from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import os
import sys
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import NLP libraries, but don't fail if they're not available
try:
    import spacy
    from spacy.tokens import Doc
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import pickle
    
    # Check if the spaCy model is installed
    def is_spacy_model_installed(model_name="en_core_web_sm"):
        try:
            spacy.load(model_name)
            return True
        except OSError:
            return False
    
    # Function to install spaCy model if not already installed
    def install_spacy_model(model_name="en_core_web_sm"):
        if not is_spacy_model_installed(model_name):
            logger.info(f"Installing spaCy model '{model_name}'...")
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                logger.info(f"Successfully installed spaCy model '{model_name}'")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install spaCy model '{model_name}': {e}")
                return False
        else:
            logger.info(f"spaCy model '{model_name}' is already installed")
            return True
    
    # Try to load the spaCy model
    if is_spacy_model_installed():
        nlp = spacy.load("en_core_web_sm")
        NLP_AVAILABLE = True
        logger.info("Successfully loaded spaCy model 'en_core_web_sm'")
    else:
        NLP_AVAILABLE = False
        logger.warning("Could not load spaCy model 'en_core_web_sm'. Please install it using: python -m spacy download en_core_web_sm")
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLP libraries not available. Using regex-only extraction.")

class SMSExtractor:
    def __init__(self, model_path: str = 'sms_extractor_model.pkl', auto_install_model: bool = False):
        """Initialize the SMS extractor with NLP models if available"""
        self.nlp = None
        self.models = {}
        
        # Initialize NLP if available
        if NLP_AVAILABLE:
            self.nlp = nlp
        elif auto_install_model:
            # Try to install the model if auto_install is enabled
            if install_spacy_model():
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Successfully loaded spaCy model after installation")
            
        # Add custom components to the pipeline if NLP is available
        if self.nlp:
            if "amount_extractor" not in self.nlp.pipe_names:
                self.nlp.add_pipe("amount_extractor", after="ner")
            if "date_extractor" not in self.nlp.pipe_names:
                self.nlp.add_pipe("date_extractor", after="ner")
            if "transaction_type_classifier" not in self.nlp.pipe_names:
                self.nlp.add_pipe("transaction_type_classifier", after="ner")
                
            # Load trained models if they exist
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models = pickle.load(f)
                    logger.info(f"Loaded models from {model_path}")
                except Exception as e:
                    logger.error(f"Error loading models: {e}")
        
        # Initialize regex patterns for fallback
        self._init_regex_patterns()
    
    def _init_regex_patterns(self):
        """Initialize regex patterns as fallback"""
        self.regex_patterns = {
            'amount': [
                r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'(\d+(?:,\d+)*(?:\.\d{2})?)\s*(?:Rs\.?|INR)',
                r'debited by\s+(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'credited by\s+Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'debited with\s+Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'credited with\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'deposited to\s+A/c\s+.*?\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'debited from\s+A/c\s+.*?\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'used for\s+Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'paid thru\s+A/C\s+.*?\s+Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'transaction number.*?for\s+Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'An amount of\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'has been\s+DEBITED\s+.*?\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'has been\s+CREDITED\s+.*?\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'debited by\s+(\d+(?:,\d+)*(?:\.\d{0,2})?)',  # Handle amounts without decimals
                r'credited by\s+Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{0,2})?)'  # Handle amounts without decimals
            ],
            'date': [
                r'(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
                r'(\d{1,2}/\d{1,2}/\d{4})',
                r'(\d{1,2}/\d{1,2}/\d{2})',  # DD/MM/YY format
                r'(\d{1,2}-\d{1,2}-\d{4})',
                r'(\d{1,2}-\d{1,2}-\d{2})',  # DD-MM-YY format
                r'date\s+(\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2})',
                r'on\s+(\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2})',
                r'on\s+(\d{1,2}-\d{1,2}-\d{4})',
                r'on\s+(\d{1,2}-\d{1,2}-\d{2})',  # on DD-MM-YY format
                r'(\d{1,2}(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d{2})',
                r'(\d{1,2}-[A-Z]{3}-\d{4})',  # DD-MMM-YYYY format
                r'(\d{1,2}-[A-Z]{3}-\d{2})',  # DD-MMM-YY format
                r'(\d{1,2}[A-Z]{3}\d{2})',  # DDMMMYY format
                r'on\s*(\d{1,2}[A-Z]{3}\d{2})',  # on DDMMMYY format
                r'Value\s+(\d{1,2}-[A-Z]{3}-\d{4})',  # Value DD-MMM-YYYY format
                r'on\s*(\d{1,2}[A-Z]{3}\d{2})\s',  # on DDMMMYY format with space
                r'on\s*(\d{1,2}[A-Z]{3}\d{2})\s*at',  # on DDMMMYY format with at
                r'on\s*(\d{1,2}[A-Z]{3}\d{2})\s*trf',  # on DDMMMYY format with trf
                r'on\s*(\d{1,2}[A-Z]{3}\d{2})\s*\.',  # on DDMMMYY format with period
                r'on\s*(\d{1,2}[A-Z]{3}\d{2})\s*$'  # on DDMMMYY format at end
            ],
            'time': [
                r'(\d{1,2}:\d{2}:\d{2})',
                r'(\d{1,2}:\d{2})',
                r'at\s+(\d{1,2}:\d{2}:\d{2})',
                r'at\s+(\d{1,2}:\d{2})',
                r'on.*?at\s+(\d{1,2}:\d{2}:\d{2})',
                r'on.*?at\s+(\d{1,2}:\d{2})',
                r'at\s+(?:\d+\s+)?(\d{1,2}:\d{2}:\d{2})',  # Handle "at 20126180 12:26:33" format
                r'at\s+(?:\d+\s+)?(\d{1,2}:\d{2})',  # Handle "at 20126180 12:26" format
                r'at\s*(\d{1,2}:\d{2}:\d{2})\s*\.',  # Handle time at end of message
                r'at\s*(\d{1,2}:\d{2})\s*\.',  # Handle time at end of message
                r'at\s*(\d{1,2}:\d{2}:\d{2})\s*$',  # Handle time at end of message
                r'at\s*(\d{1,2}:\d{2})\s*$'  # Handle time at end of message
            ],
            'transaction_id': [
                r'(?:Txn|Transaction)\s*(?:ID|id)?\s*[:#]?\s*([A-Z0-9]+)',
                r'(?:Ref|Reference)\s*(?:No|no)?\s*[:#]?\s*([A-Z0-9]+)',
                r'Refno\s+(\d+)',
                r'Ref\s+No\s+(\d+)',
                r'Txn#\s*(\d+)',
                r'transaction number\s+(\d+)',
                r'UPI Ref\s+(\d+)',
                r'Ref\.No\s*:?\s*(\d+)',
                r'UPI/DR/([A-Z0-9]+)/',  # Handle UPI/DR/ID format
                r'at\s*(\d{8})\s',  # Handle merchant ID format
                r'Value\s+\d{2}-[A-Z]{3}-\d{4}\s*\.\s*Clear',  # Handle Value date format
                r'Refno\s+(\d+)\s*\.',  # Handle Refno at end of message
                r'Ref\.No\s*:?\s*(\d+)\s*\.',  # Handle Ref.No at end of message
                r'Ref\.No\s*:?\s*(\d+)\s*$',  # Handle Ref.No at end of message
                r'Refno\s+(\d+)\s*$'  # Handle Refno at end of message
            ],
            'account_number': [
                r'(?:Acc|Account)\s*(?:No|no)?\s*[:#]?\s*([A-Z0-9]+)',
                r'account\s+XXX(\d+)',
                r'XXX(\d+)',
                r'A/C\s+[X]*(\d+)',
                r'account\s+[X]*(\d+)',
                r'A/c\s+XXXXXXXXXX(\d+)',
                r'A/c\s+[X]*(\d+)',
                r'Card\s+[X]*(\d+)',
                r'Debit Card\s+[X]*(\d+)',
                r'SBIDrCard\s+[X]*(\d+)',
                r'A/C\s+XX(\d+)',
                r'A/c\s+X(\d+)',
                r'your\s+A/c\s+[X]*(\d+)'  # Handle "your A/c X1234" format
            ],
            'bank_name': [
                r'(?:from|to)\s+([A-Za-z\s]+(?:Bank|BANK))',
                r'([A-Za-z\s]+(?:Bank|BANK))\s+(?:has|on|at)',
                r'([A-Za-z\s]+(?:Bank|BANK))\s*$',
                r'-\s*([A-Za-z\s]+(?:Bank|BANK))',
                r'-\s*([A-Z]{2,})',
                r'([A-Za-z\s]+(?:Bank|BANK))\s+(?:towards|Clear)',
                r'([A-Za-z\s]+(?:Bank|BANK))',
                r'(?:Canara|SBI|Bandhan)\s*Bank',
                r'-\s*(SBI)'  # Handle "-SBI" format
            ],
            'upi_id': [
                r'([a-zA-Z0-9._-]+@[a-zA-Z]{3,})',
                r'UPI\s*ID\s*[:#]?\s*([a-zA-Z0-9._-]+)',
                r'trf to\s+([A-Z0-9\s]+)',
                r'to\s+([A-Z\s]+(?:LIMITED|CLINIC))',
                r'to\s+([A-Z\s]+)',
                r'transfer from\s+([A-Z\s]+)',
                r'trf to\s+([A-Z\s]+)',
                r'UPI/DR/[A-Z0-9]+/([A-Z\s]+)',
                r'(?:to|from)\s+([A-Z\s]+(?:LIMITED|CLINIC|MAZUMDER))',  # Handle specific merchant names
                r'trf to\s+([A-Z\s]+(?:LIMITED|CLINIC|MAZUMDER))',  # Handle specific merchant names
                r'to\s+([A-Z\s]+(?:LIMITED|CLINIC|MAZUMDER))\s*\.',  # Handle merchant names at end
                r'trf to\s+([A-Z\s]+(?:LIMITED|CLINIC|MAZUMDER))\s*\.',  # Handle merchant names at end
                r'to\s+([A-Z\s]+(?:LIMITED|CLINIC|MAZUMDER))\s*$',  # Handle merchant names at end
                r'trf to\s+([A-Z\s]+(?:LIMITED|CLINIC|MAZUMDER))\s*$'  # Handle merchant names at end
            ],
            'transaction_type': [
                r'(?:credited|debited|spent|received|paid)',
                r'(?:CREDIT|DEBIT|SPENT|RECEIVED|PAID)',
                r'debited by',
                r'credited by',
                r'debited with',
                r'credited with',
                r'deposited to',
                r'debited from',
                r'used for',
                r'paid thru',
                r'has been\s+DEBITED',
                r'has been\s+CREDITED',
                r'transfer from',  # Handle "transfer from" format
                r'trf to'  # Handle "trf to" format
            ],
            'balance': [
                r'(?:balance|bal)\s*(?:is|:)?\s*Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'(?:balance|bal)\s*(?:is|:)?\s*INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'(?:Avail\.?bal|Available\s+balance)\s*(?:is|:)?\s*INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'(?:Avail\.?bal|Available\s+balance)\s*(?:is|:)?\s*Rs\.?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'Clear Bal is\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'Total Avail\.?bal\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                r'Clear Bal is\s+INR\s*(\d+(?:,\d+)*(?:\.\d{2})?)\s*\.'  # Handle balance at end of message
            ]
        }
    
    def extract_transaction_data(self, message: str) -> Dict[str, Any]:
        """Extract transactional data from a single SMS message using NLP and regex"""
        data = {}
        
        # Process the message with spaCy if available
        doc = None
        if self.nlp:
            doc = self.nlp(message)
        
        # Extract amount using NLP or regex
        amount = None
        if doc:
            amount = self._extract_amount_nlp(doc)
        if amount is not None:
            data['amount'] = amount
        else:
            # Fallback to regex
            for pattern in self.regex_patterns['amount']:
                amount_match = re.search(pattern, message)
                if amount_match:
                    try:
                        amount_str = amount_match.group(1).replace(',', '')
                        data['amount'] = float(amount_str)
                        break
                    except (ValueError, AttributeError):
                        continue
        
        # Extract date using NLP or regex
        date = None
        if doc:
            date = self._extract_date_nlp(doc)
        if date is not None:
            data['date'] = date
        else:
            # Fallback to regex
            for pattern in self.regex_patterns['date']:
                date_match = re.search(pattern, message)
                if date_match:
                    date_str = date_match.group(1)
                    try:
                        # Handle different date formats
                        if '/' in date_str:
                            if len(date_str.split('/')[2]) == 2:
                                data['date'] = datetime.strptime(date_str, '%d/%m/%y').strftime('%Y-%m-%d')
                            else:
                                data['date'] = datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
                        elif '-' in date_str:
                            parts = date_str.split('-')
                            if len(parts) == 3:
                                if len(parts[2]) == 2:  # YY format
                                    if parts[1].isalpha():  # DD-MMM-YY
                                        data['date'] = datetime.strptime(date_str, '%d-%b-%y').strftime('%Y-%m-%d')
                                    else:  # DD-MM-YY
                                        data['date'] = datetime.strptime(date_str, '%d-%m-%y').strftime('%Y-%m-%d')
                                else:  # YYYY format
                                    if parts[1].isalpha():  # DD-MMM-YYYY
                                        data['date'] = datetime.strptime(date_str, '%d-%b-%Y').strftime('%Y-%m-%d')
                                    else:  # DD-MM-YYYY
                                        data['date'] = datetime.strptime(date_str, '%d-%m-%Y').strftime('%Y-%m-%d')
                        elif len(date_str) == 7 and date_str[2:5].isalpha():  # Format like 17Mar25
                            data['date'] = datetime.strptime(date_str, '%d%b%y').strftime('%Y-%m-%d')
                        elif len(date_str) == 9 and 'on' not in date_str:  # Format like 10Dec2020
                            data['date'] = datetime.strptime(date_str, '%d%b%Y').strftime('%Y-%m-%d')
                        else:
                            # Try different date formats
                            for fmt in ['%d %B %Y', '%d %b %Y', '%d-%b-%Y', '%d%B%y', '%d%b%y']:
                                try:
                                    data['date'] = datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                                    break
                                except ValueError:
                                    continue
                        break
                    except ValueError:
                        continue
        
        # Extract time using NLP or regex
        time = None
        if doc:
            time = self._extract_time_nlp(doc)
        if time is not None:
            data['time'] = time
        else:
            # Fallback to regex
            for pattern in self.regex_patterns['time']:
                time_match = re.search(pattern, message)
                if time_match:
                    time_str = time_match.group(1)
                    # Standardize time format
                    try:
                        if ':' in time_str:
                            parts = time_str.split(':')
                            if len(parts) == 2:
                                data['time'] = f"{int(parts[0]):02d}:{int(parts[1]):02d}"
                            elif len(parts) == 3:
                                data['time'] = f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Extract transaction ID using NLP or regex
        txn_id = None
        if doc:
            txn_id = self._extract_transaction_id_nlp(doc)
        if txn_id is not None:
            data['transaction_id'] = txn_id
        else:
            # Fallback to regex
            for pattern in self.regex_patterns['transaction_id']:
                txn_id_match = re.search(pattern, message)
                if txn_id_match:
                    data['transaction_id'] = txn_id_match.group(1).strip()
                    break
        
        # Extract account number using NLP or regex
        acc_num = None
        if doc:
            acc_num = self._extract_account_number_nlp(doc)
        if acc_num is not None:
            data['account_number'] = acc_num
        else:
            # Fallback to regex
            for pattern in self.regex_patterns['account_number']:
                acc_match = re.search(pattern, message)
                if acc_match:
                    acc_num = acc_match.group(1).strip()
                    # Clean up account number
                    acc_num = re.sub(r'[^0-9]', '', acc_num)
                    if acc_num:
                        data['account_number'] = acc_num
                        break
        
        # Extract bank name using NLP or regex
        bank_name = None
        if doc:
            bank_name = self._extract_bank_name_nlp(doc)
        if bank_name is not None:
            data['bank_name'] = bank_name
        else:
            # Fallback to regex
            for pattern in self.regex_patterns['bank_name']:
                bank_match = re.search(pattern, message)
                if bank_match:
                    bank_name = bank_match.group(1).strip() if bank_match.groups() else bank_match.group(0).strip()
                    # Clean up bank name
                    bank_name = re.sub(r'\s+', ' ', bank_name)
                    if bank_name:
                        data['bank_name'] = bank_name
                        break
        
        # Extract UPI ID using NLP or regex
        upi_id = None
        if doc:
            upi_id = self._extract_upi_id_nlp(doc)
        if upi_id is not None:
            data['upi_id'] = upi_id
        else:
            # Fallback to regex
            for pattern in self.regex_patterns['upi_id']:
                upi_match = re.search(pattern, message)
                if upi_match:
                    upi_id = upi_match.group(1).strip()
                    # Clean up UPI ID
                    upi_id = re.sub(r'\s+', ' ', upi_id)
                    # Remove trailing 'R' if present
                    if upi_id.endswith(' R'):
                        upi_id = upi_id[:-2]
                    # Remove trailing 'V' if present
                    if upi_id.endswith(' V'):
                        upi_id = upi_id[:-2]
                    if upi_id:
                        data['upi_id'] = upi_id
                        break
        
        # Determine transaction type using NLP or regex
        txn_type = None
        if doc:
            txn_type = self._extract_transaction_type_nlp(doc)
        if txn_type is not None:
            data['transaction_type'] = txn_type
        else:
            # Fallback to regex
            message_lower = message.lower()
            if any(word in message_lower for word in ['credited', 'received', 'credit', 'deposited']):
                data['transaction_type'] = 'credit'
            elif any(word in message_lower for word in ['debited', 'spent', 'paid', 'debit', 'used']):
                data['transaction_type'] = 'debit'
        
        # Extract balance using NLP or regex
        balance = None
        if doc:
            balance = self._extract_balance_nlp(doc)
        if balance is not None:
            data['balance'] = balance
        else:
            # Fallback to regex
            for pattern in self.regex_patterns['balance']:
                balance_match = re.search(pattern, message)
                if balance_match:
                    try:
                        balance_str = balance_match.group(1).replace(',', '')
                        data['balance'] = float(balance_str)
                        break
                    except (ValueError, AttributeError):
                        continue
        
        return data
    
    def _extract_amount_nlp(self, doc: Doc) -> Optional[float]:
        """Extract amount using NLP"""
        # Look for currency entities
        for ent in doc.ents:
            if ent.label_ in ['MONEY', 'CARDINAL']:
                try:
                    # Clean up the amount string
                    amount_str = ent.text.replace('Rs.', '').replace('INR', '').replace(',', '').strip()
                    return float(amount_str)
                except (ValueError, AttributeError):
                    continue
        
        # Look for specific patterns in the text
        for token in doc:
            if token.like_num and token.i < len(doc) - 1:
                next_token = doc[token.i + 1]
                if next_token.text in ['Rs.', 'INR']:
                    try:
                        amount_str = token.text.replace(',', '')
                        return float(amount_str)
                    except (ValueError, AttributeError):
                        continue
        
        return None
    
    def _extract_date_nlp(self, doc: Doc) -> Optional[str]:
        """Extract date using NLP"""
        # Look for date entities
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                try:
                    # Try to parse the date
                    date_str = ent.text
                    # Handle different date formats
                    if '/' in date_str:
                        if len(date_str.split('/')[2]) == 2:
                            return datetime.strptime(date_str, '%d/%m/%y').strftime('%Y-%m-%d')
                        else:
                            return datetime.strptime(date_str, '%d/%m/%Y').strftime('%Y-%m-%d')
                    elif '-' in date_str:
                        parts = date_str.split('-')
                        if len(parts) == 3:
                            if len(parts[2]) == 2:  # YY format
                                if parts[1].isalpha():  # DD-MMM-YY
                                    return datetime.strptime(date_str, '%d-%b-%y').strftime('%Y-%m-%d')
                                else:  # DD-MM-YY
                                    return datetime.strptime(date_str, '%d-%m-%y').strftime('%Y-%m-%d')
                            else:  # YYYY format
                                if parts[1].isalpha():  # DD-MMM-YYYY
                                    return datetime.strptime(date_str, '%d-%b-%Y').strftime('%Y-%m-%d')
                                else:  # DD-MM-YYYY
                                    return datetime.strptime(date_str, '%d-%m-%Y').strftime('%Y-%m-%d')
                        elif len(date_str) == 7 and date_str[2:5].isalpha():  # Format like 17Mar25
                            return datetime.strptime(date_str, '%d%b%y').strftime('%Y-%m-%d')
                        elif len(date_str) == 9 and 'on' not in date_str:  # Format like 10Dec2020
                            return datetime.strptime(date_str, '%d%b%Y').strftime('%Y-%m-%d')
                        else:
                            # Try different date formats
                            for fmt in ['%d %B %Y', '%d %b %Y', '%d-%b-%Y', '%d%B%y', '%d%b%y']:
                                try:
                                    return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
                                except ValueError:
                                    continue
                except ValueError:
                    continue
        
        # Look for specific patterns in the text
        for i, token in enumerate(doc):
            if token.text.lower() in ['on', 'date'] and i < len(doc) - 1:
                next_token = doc[i + 1]
                if next_token.like_num:
                    try:
                        # Try to parse the date
                        date_str = next_token.text
                        if i < len(doc) - 2:
                            month_token = doc[i + 2]
                            if month_token.text in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                                date_str += ' ' + month_token.text
                                if i < len(doc) - 3:
                                    year_token = doc[i + 3]
                                    if year_token.like_num:
                                        date_str += ' ' + year_token.text
                                        try:
                                            return datetime.strptime(date_str, '%d %b %Y').strftime('%Y-%m-%d')
                                        except ValueError:
                                            try:
                                                return datetime.strptime(date_str, '%d %b %y').strftime('%Y-%m-%d')
                                            except ValueError:
                                                pass
                    except (ValueError, AttributeError):
                        continue
        
        return None
    
    def _extract_time_nlp(self, doc: Doc) -> Optional[str]:
        """Extract time using NLP"""
        # Look for time entities
        for ent in doc.ents:
            if ent.label_ == 'TIME':
                try:
                    # Try to parse the time
                    time_str = ent.text
                    if ':' in time_str:
                        parts = time_str.split(':')
                        if len(parts) == 2:
                            return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
                        elif len(parts) == 3:
                            return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
                except (ValueError, AttributeError):
                    continue
        
        # Look for specific patterns in the text
        for i, token in enumerate(doc):
            if token.text.lower() == 'at' and i < len(doc) - 1:
                next_token = doc[i + 1]
                if ':' in next_token.text:
                    try:
                        time_str = next_token.text
                        parts = time_str.split(':')
                        if len(parts) == 2:
                            return f"{int(parts[0]):02d}:{int(parts[1]):02d}"
                        elif len(parts) == 3:
                            return f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(parts[2]):02d}"
                    except (ValueError, AttributeError):
                        continue
        
        return None
    
    def _extract_transaction_id_nlp(self, doc: Doc) -> Optional[str]:
        """Extract transaction ID using NLP"""
        # Look for specific patterns in the text
        for i, token in enumerate(doc):
            if token.text.lower() in ['ref', 'refno', 'ref.no', 'txn', 'transaction'] and i < len(doc) - 1:
                next_token = doc[i + 1]
                if next_token.like_num:
                    return next_token.text
                elif next_token.text == ':' and i < len(doc) - 2:
                    next_next_token = doc[i + 2]
                    if next_next_token.like_num:
                        return next_next_token.text
        
        return None
    
    def _extract_account_number_nlp(self, doc: Doc) -> Optional[str]:
        """Extract account number using NLP"""
        # Look for specific patterns in the text
        for i, token in enumerate(doc):
            if token.text.lower() in ['a/c', 'account', 'acc'] and i < len(doc) - 1:
                next_token = doc[i + 1]
                if 'X' in next_token.text and any(c.isdigit() for c in next_token.text):
                    # Extract digits from the account number
                    acc_num = ''.join(c for c in next_token.text if c.isdigit())
                    if acc_num:
                        return acc_num
        
        return None
    
    def _extract_bank_name_nlp(self, doc: Doc) -> Optional[str]:
        """Extract bank name using NLP"""
        # Look for specific patterns in the text
        for i, token in enumerate(doc):
            if token.text.lower() == 'bank' and i > 0:
                prev_token = doc[i - 1]
                if prev_token.text in ['Canara', 'SBI', 'Bandhan']:
                    return f"{prev_token.text} Bank"
        
        # Look for bank names at the end of the message
        if len(doc) > 1:
            last_token = doc[-1]
            if last_token.text == 'Bank':
                prev_token = doc[-2]
                return f"{prev_token.text} Bank"
        
        return None
    
    def _extract_upi_id_nlp(self, doc: Doc) -> Optional[str]:
        """Extract UPI ID using NLP"""
        # Look for specific patterns in the text
        for i, token in enumerate(doc):
            if token.text.lower() in ['to', 'from', 'trf'] and i < len(doc) - 1:
                next_token = doc[i + 1]
                if next_token.text.isupper():
                    upi_id = next_token.text
                    # Check if there are more tokens that might be part of the UPI ID
                    if i < len(doc) - 2:
                        next_next_token = doc[i + 2]
                        if next_next_token.text.isupper():
                            upi_id += ' ' + next_next_token.text
                    # Clean up UPI ID
                    upi_id = re.sub(r'\s+', ' ', upi_id)
                    # Remove trailing 'R' if present
                    if upi_id.endswith(' R'):
                        upi_id = upi_id[:-2]
                    # Remove trailing 'V' if present
                    if upi_id.endswith(' V'):
                        upi_id = upi_id[:-2]
                    return upi_id
        
        return None
    
    def _extract_transaction_type_nlp(self, doc: Doc) -> Optional[str]:
        """Extract transaction type using NLP"""
        # Look for specific patterns in the text
        for token in doc:
            if token.text.lower() in ['credited', 'received', 'credit', 'deposited']:
                return 'credit'
            elif token.text.lower() in ['debited', 'spent', 'paid', 'debit', 'used']:
                return 'debit'
        
        return None
    
    def _extract_balance_nlp(self, doc: Doc) -> Optional[float]:
        """Extract balance using NLP"""
        # Look for specific patterns in the text
        for i, token in enumerate(doc):
            if token.text.lower() in ['balance', 'bal'] and i < len(doc) - 2:
                next_token = doc[i + 1]
                if next_token.text.lower() in ['is', ':'] and i < len(doc) - 2:
                    next_next_token = doc[i + 2]
                    if next_next_token.text in ['Rs.', 'INR'] and i < len(doc) - 3:
                        amount_token = doc[i + 3]
                        if amount_token.like_num:
                            try:
                                amount_str = amount_token.text.replace(',', '')
                                return float(amount_str)
                            except (ValueError, AttributeError):
                                pass
        
        return None
    
    def process_messages(self, messages: List[str]) -> pd.DataFrame:
        """Process a list of SMS messages and return a DataFrame"""
        transactions = []
        
        for message in messages:
            data = self.extract_transaction_data(message)
            if data:  # Only add if some data was extracted
                transactions.append(data)
        
        return pd.DataFrame(transactions)
    
    def save_to_csv(self, df: pd.DataFrame, filename: str = 'transactions.csv'):
        """Save the extracted data to a CSV file"""
        try:
            df.to_csv(filename, index=False)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    def save_to_json(self, df: pd.DataFrame, filename: str = 'transactions.json'):
        """Save the extracted data to a JSON file"""
        try:
            df.to_json(filename, orient='records', indent=4)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving to JSON: {e}")
    
    def train_models(self, training_data: List[Dict[str, Any]], model_path: str = 'sms_extractor_model.pkl'):
        """Train NLP models on labeled data"""
        if not NLP_AVAILABLE:
            logger.warning("NLP libraries not available. Cannot train models.")
            return
            
        # This is a placeholder for training models on labeled data
        # In a real implementation, you would train models for each extraction task
        logger.info("Training models on labeled data...")
        
        # Example: Train a transaction type classifier
        if 'transaction_type' in training_data[0]:
            X = [item['message'] for item in training_data]
            y = [item['transaction_type'] for item in training_data]
            
            vectorizer = TfidfVectorizer(max_features=1000)
            X_vec = vectorizer.fit_transform(X)
            
            model = LogisticRegression()
            model.fit(X_vec, y)
            
            self.models['transaction_type'] = {
                'vectorizer': vectorizer,
                'model': model
            }
        
        # Save the trained models
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.models, f)
            logger.info(f"Models saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

def main():
    # Example usage
    extractor = SMSExtractor()
    
    # Example messages
    sample_messages = [
        "   An amount of INR 5.00 has been DEBITED to your account XXX670 on 12/03/2025. Total Avail.bal INR 3,127.41.Dial 1930 to report cyber fraud - Canara Bank",   
        "Dear customer, Your account XXXXXXXXXX1234 is debited with Rs.1234.00 on 27/Aug/2024 19:59:09. Ref.No:123456789012",
        "Dear UPI user A/C X7712 debited by 265.94 on date 17Mar25 trf to BBNOW Refno 544272365695. If not u? call 1800111109. -SBI",
        "    Rs.20.00 paid thru A/C XX7670 on 12-2-25 10:46:11 to IRCTC UTS, UPI Ref 504363964408. If not done, SMS BLOCKUPI to 9901771222.-Canara Bank",
        "Dear Customer, your account XXXXXXXXXX1234 is credited with INR 12.00 on 01-JAN-2025 towards interest. Bandhan Bank",
        "SBIDrCard X3177 used for Rs499.00 on10Dec20 at20126180 Txn#002961255575 If not done fwd this SMS to 9223008333/call1800111109/9449112211 to block Card",
        "INR 1234.00 deposited to A/c XXXXXXXXXX1234 towards NEFT Cr-ICIC0000001-IDIGIPAY-C on 04-SEP-2024 . Clear Bal is INR 1,234.00 . Bandhan Bank",
        "Dear customer, transaction number 135206974419 for Rs1943.00 by SBI Debit Card X3177 at 87062596 on 19Dec21 at 12:26:33. If not done forward this SMS to 9223008333 or call 18001111109/9449112211 to block card",
        "INR 1234.00 debited from A/c XXXXXXXXXX1601 towards UPI/DR/D123456789012/HIRANMOY Value 09-DEC-2024 . Clear Bal is INR 1234.00. Bandhan Bank",
        "Dear UPI user A/C X7712 debited by 2575.0 on date 29Jul24 trf to SOURAV MAZUMDER Refno 421187756511. If not u? call 1800111109. -SBI",
        "Dear SBI User, your A/c X7712-credited by Rs.500 on 13Mar25 transfer from SOURAV MAZUMDER Ref No 543805285902 -SBI",
        "Dear UPI user A/C X7712 debited by 1000.0 on date 05Mar25 trf to ABHI CLINIC Refno 506464754574. If not u? call 1800111109. -SBI",
        "Dear UPI user A/C X7712 debited by 226.15 on date 18Nov24 trf to ZOMATO LIMITED Refno 432315651473. If not u? call 1800111109. -SBI",
    ]
    
    try:
        # Process messages
        df = extractor.process_messages(sample_messages)
        
        # Save to files
        extractor.save_to_csv(df)
        extractor.save_to_json(df)
        
        # Display results
        logger.info("\nExtracted Transactions:")
        print(df)
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 
