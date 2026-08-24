# Windows quick start

1. Open PowerShell in the project folder.
2. Create environment:
   `python -m venv .venv`
3. Activate:
   `.venv\Scripts\Activate.ps1`
4. Install:
   `pip install -r requirements.txt`
5. Create `.env` from `.env.example`.
6. Enter the MySQL password.
7. Run:
   `python -m src.db_check`
8. Then:
   `python -m src.eda`
9. Then:
   `python -m src.train_model`
10. Test:
   `python -m src.assess_demo`
11. Start API:
   `uvicorn src.api:app --reload`
12. In a second terminal:
   `streamlit run app.py`

If PowerShell blocks activation:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
