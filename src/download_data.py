"""
Lloyd Onny — 10211100341

Download assignment datasets (Ghana election CSV + 2025 budget PDF).
"""

import requests

# Download CSV
def download_csv():
    url = "https://github.com/GodwinDansoAcity/acitydataset/raw/main/Ghana_Election_Result.csv"
    r = requests.get(url)
    with open("data/Ghana_Election_Result.csv", "wb") as f:
        f.write(r.content)

# Download PDF
def download_pdf():
    url = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    r = requests.get(url)
    with open("data/2025-Budget-Statement-and-Economic-Policy_v4.pdf", "wb") as f:
        f.write(r.content)

if __name__ == "__main__":
    download_csv()
    download_pdf()
    print("Datasets downloaded to data/ directory.")
