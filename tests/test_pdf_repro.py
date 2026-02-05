
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.prediction_logic import generate_report_pdf

# Test data with potential problematic characters
prediction_val = 123.456
features = {
    'PrimaryPropertyType': 'Office',
    'PropertyGFATotal': 50000,
    'ENERGYSTARScore': 75,
    'Neighborhood': 'Downtown',
    'YearBuilt': 1990,
    'NumberofFloors': 10
}

try:
    print("Attempting to generate PDF...")
    pdf_bytes = generate_report_pdf(prediction_val, features)
    if pdf_bytes:
        print(f"Success! Generated {len(pdf_bytes)} bytes.")
        with open("test_report.pdf", "wb") as f:
            f.write(pdf_bytes)
        print("Written to test_report.pdf")
    else:
        print("Failed: generate_report_pdf returned None")

    # Test with special characters
    features['Neighborhood'] = "Café & Château €"
    print("\nAttempting with special characters...")
    pdf_bytes = generate_report_pdf(prediction_val, features)
    if pdf_bytes:
        print(f"Success with special chars! Generated {len(pdf_bytes)} bytes.")
    else:
        print("Failed with special chars")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
