# 🤖 AI vs Human Text Classifier Bot
Fine-tuned BERT for text classification task (AI-generated vs. human-written) with a Telegram bot interface  
### Telegram link: https://t.me/AI_or_Human_bot   
### Enjoy using it✨

# 🛠️ Technical Stack   
### Core Model: bert-base-multilingual-uncased fine-tuned on 1.3M texts   
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
# Datasets on which the model was trained:
https://www.kaggle.com/datasets/starblasters8/human-vs-llm-text-corpus - 800K texts   
https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text - 500K texts   
