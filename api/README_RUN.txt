PHISHING WEBSITE DETECTION - LIVE DEMO

1. Open this folder in Command Prompt.
2. Install packages:
   pip install -r requirements.txt
3. Run:
   streamlit run app.py
4. Browser opens automatically. If not, open the Local URL shown by Streamlit.
5. Enter a website URL and click Check Website.

WHAT THIS VERSION DOES
- Accepts a normal website URL.
- Fetches the webpage with a short timeout and without executing JavaScript.
- Extracts URL and HTML/JavaScript-related signals.
- Uses the same 30-feature dataset schema for the XGBoost deployment model.
- Shows a clear final verdict: PHISHING or LEGITIMATE.
- Shows the phishing-class model score as supporting information.

IMPORTANT RESEARCH NOTE
The original research experiment is not overwritten. The live deployment is a practical extension. Several original dataset fields require WHOIS/reputation/search-engine data (domain age, traffic, PageRank, Google index, inbound links), so the live app does not pretend to have measured them. Those fields use neutral fallback values. This means the live detector should be presented as a research/demo deployment, not as a guaranteed real-time security service.
