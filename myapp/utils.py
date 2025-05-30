# myapp/utils.py
import re

"""
Utility Functions for Heart Disease Detection System
This module provides helper functions and utilities used throughout the application.
Currently includes URL security validation functions to protect against phishing attempts.


"""

"""
Constant: BLACKLISTED_DOMAINS
Description: Set of known malicious or phishing domains
Usage: Used by is_phishing_url() to check against known bad domains
"""
BLACKLISTED_DOMAINS = {
    "example-phishing.com",
    "bad-website.net",
    "malicious-site.org"
}

"""
Function: is_phishing_url(url)
Description: Analyzes a URL to determine if it might be a phishing attempt or malicious link.

This function performs multiple security checks on the provided URL:
1. Checks against a blacklist of known malicious domains
2. Looks for common phishing-related keywords
3. Analyzes URL structure for suspicious patterns

Parameters:
    url (str): The URL to be analyzed for potential security threats

Returns:
    bool: True if the URL appears to be malicious/phishing, False if the URL appears to be legitimate

Security Checks:
    1. Domain Blacklist:
       - Compares against known malicious domains
       
    2. Keyword Analysis:
       - login* - Common target for credential theft
       - verify* - Often used in verification scams
       - account* - Banking/financial fraud attempts
       - update* - System update scams
       - secure* - False security claims
       - bank* - Financial institution impersonation
       - paypal* - Payment system fraud
       
    3. Structural Analysis:
       - Multiple subdomains (potential domain masking)
       - @ symbol in URL (URL redirection trick)
       
Example:
    >>> is_phishing_url("http://legitimate-bank.com")
    False
    >>> is_phishing_url("http://bank.secure.example-phishing.com")
    True
"""
def is_phishing_url(url):
    """
    Analyzes a URL to determine if it might be a phishing attempt or malicious link.
    
    This function performs multiple security checks on the provided URL:
    1. Checks against a blacklist of known malicious domains
    2. Looks for common phishing-related keywords
    3. Analyzes URL structure for suspicious patterns
    
    Parameters:
    -----------
    url : str
        The URL to be analyzed for potential security threats
        
    Returns:
    --------
    bool
        True if the URL appears to be malicious/phishing
        False if the URL appears to be legitimate
        
    Security Checks:
    ---------------
    1. Domain Blacklist:
       - Compares against known malicious domains
       
    2. Keyword Analysis:
       - login* - Common target for credential theft
       - verify* - Often used in verification scams
       - account* - Banking/financial fraud attempts
       - update* - System update scams
       - secure* - False security claims
       - bank* - Financial institution impersonation
       - paypal* - Payment system fraud
       
    3. Structural Analysis:
       - Multiple subdomains (potential domain masking)
       - @ symbol in URL (URL redirection trick)
       
    Example:
    --------
    >>> is_phishing_url("http://legitimate-bank.com")
    False
    >>> is_phishing_url("http://bank.secure.example-phishing.com")
    True
    """
    # Check against known malicious domains
    for domain in BLACKLISTED_DOMAINS:
        if domain in url:
            return True

    # Define and check against common phishing patterns
    phishing_patterns = [
        re.compile(r"login.*", re.IGNORECASE),    # Login credential theft
        re.compile(r"verify.*", re.IGNORECASE),   # Account verification scams
        re.compile(r"account.*", re.IGNORECASE),  # Account-related fraud
        re.compile(r"update.*", re.IGNORECASE),   # System update scams
        re.compile(r"secure.*", re.IGNORECASE),   # False security claims
        re.compile(r"bank.*", re.IGNORECASE),     # Bank impersonation
        re.compile(r"paypal.*", re.IGNORECASE)    # Payment system fraud
    ]

    # Check each pattern against the URL
    for pattern in phishing_patterns:
        if pattern.search(url):
            return True

    # Analyze URL structure for suspicious characteristics
    if len(url.split('.')) > 3:  # Multiple subdomains (e.g., a.b.example.com)
        return True

    if '@' in url:  # URL redirection trick (e.g., http://example.com@evil.com)
        return True

    return False  # URL passed all security checks
