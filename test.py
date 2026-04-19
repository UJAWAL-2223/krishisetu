import requests, gzip

r = requests.get(
    "https://www.myscheme.gov.in/sitemap-0.xml",
    headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "gzip, deflate"},
)
print(r.status_code)
print(r.content[:1000])  # raw bytes
print("---")
print(r.text[:1000])     # decoded text