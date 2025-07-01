# 🤖 AI vs Human Text Classifier Bot
Fine-tuned BERT for text classification task (AI-generated vs. human-written) with a Telegram bot interface   

# 🛠️ Technical Stack   
### Core Model: bert-base-multilingual-uncased fine-tuned on 1 million samples   
### Backend: Python + PyTorch + Transformers   
### Telegram API: python-telegram-bot   
### Deployment: Docker + Yandex Cloud Serverless Containers   
```
ai_detect_bot/   
├── 📁 my_finetuned_bert/    # BERT weights      
│   
├── 📄 bot.py                # 🧠 Main bot logic   
├── 📄 data_loading.py       # 💾 Loading datasets from kaggle        
├── 📄 train_model.py        # ✨ BERT Fine-Tuning     
├── 📄 text_classify.py      # 🤖 Text classifying      
│   
├── 📄 .env.                 # ⚙️ Environment   
└── 📄 requirements.txt      # 📦 Dependencies
```
