# Streamlit Cloud Deployment Guide

## ⚠️ Pre-Deployment Checklist

### 1. Build Indices Locally
Before deploying, you **must** build the FAISS and TF-IDF indices:

```bash
python src/embedding_index.py
```

This creates:
- `data/faiss_index.bin` (dense vector index)
- `data/tfidf_keyword_index.joblib` (sparse keyword index)

### 2. Commit Indices to Git
```bash
git add data/faiss_index.bin data/tfidf_keyword_index.joblib
git commit -m "Add FAISS and TF-IDF indices for production"
git push
```

### 3. Set Secrets on Streamlit Cloud

Go to your Streamlit Cloud app settings → Secrets and add:

```toml
openai_api_key = "sk-your-actual-key-here"
```

### 4. Deploy

Push to your connected GitHub repo and Streamlit Cloud will auto-deploy.

---

## 🚀 Deployment Steps

### Option A: Streamlit Cloud (Recommended)

1. **Create GitHub repo** with this project
2. **Sign up at** streamlit.io/cloud
3. **Connect repo** to Streamlit Cloud
4. **Add secrets** in app settings
5. **Deploy** - app auto-deploys on each git push

### Option B: Docker (Self-Hosted)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "src/app.py", "--server.port=8501"]
```

Deploy with:
```bash
docker build -t rag-chatbot .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... rag-chatbot
```

---

## ❌ Troubleshooting

### App Loading Forever
**Problem:** FAISS indices missing or not committed
**Solution:** Run `python src/embedding_index.py` and commit indices

### 404 - File not found
**Problem:** `.gitignore` excluded index files
**Solution:** Remove indices from `.gitignore`, run `git add -f`, push again

### No responses from LLM
**Problem:** API key not set
**Solution:** Add `openai_api_key` to Streamlit Cloud Secrets

### "Missing FAISS index" error
**Problem:** Index files didn't upload to Streamlit
**Solution:** Verify files in git, check `.gitignore`, re-deploy

---

## ⚡ Performance Tips

- **Index files must be in git** (not in `.gitignore`)
- **Use `@st.cache_resource`** for expensive operations (already done)
- **Streamlit Cloud free tier limits:** 3 apps, 1 GB memory
- **For large indices**, consider:
  - Breaking data into smaller chunks
  - Using a database (PostgreSQL + pgvector)
  - Using external vector store (Pinecone, Weaviate)

---

## 📋 Deployment Checklist

- [ ] Run `python src/embedding_index.py` locally
- [ ] Verify `data/faiss_index.bin` exists
- [ ] Verify `data/tfidf_keyword_index.joblib` exists
- [ ] Comment out large files in `.gitignore` OR remove them
- [ ] Run `git add data/*.bin data/*.joblib`
- [ ] Commit and push to GitHub
- [ ] Create Streamlit Cloud app
- [ ] Add `openai_api_key` to Secrets
- [ ] Verify app loads without "Missing index" error
- [ ] Test a query end-to-end

---

## 📞 Support

If deployment fails:
1. Check Streamlit Cloud logs (app dashboard)
2. Verify all indices exist: `ls -la data/`
3. Check git status: `git status`
4. Re-run: `git add -A && git commit && git push`
