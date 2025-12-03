# 🔥 Inferno-AI

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Infernodude777/Inferno-AI/train.yml?branch=main&label=Training&logo=github)
![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A custom machine learning model built **from scratch** with **automated continuous training** using GitHub Actions. Train, evaluate, and deploy your own AI models for free!

## 🎯 Features

- ✅ **Custom ML Model**: RandomForest classifier built with scikit-learn
- 🔄 **Continuous Training**: Automated daily training via GitHub Actions
- 🆓 **100% Free**: Uses GitHub's free tier (2,000 minutes/month)
- 📊 **Auto-Reporting**: Metrics and visualizations generated automatically
- 💾 **Version Control**: All training results tracked in Git
- ⚡ **Easy Setup**: Clone and run - no complex infrastructure

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Infernodude777/Inferno-AI.git
cd Inferno-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train Your Model
```bash
python train.py
```

### 4. View Results
Check the `results/` folder for:
- `metrics.json` - Training accuracy and metadata
- `confusion_matrix.png` - Visual model performance

## 🤖 How It Works

### Architecture
```
Inferno-AI/
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
├── .github/workflows/
│   └── train.yml         # Automated CI/CD pipeline
├── models/               # Saved model files
└── results/              # Training metrics & plots
```

### Training Pipeline
1. **Data Generation**: Creates synthetic classification dataset
2. **Model Training**: Trains RandomForest with 100 estimators
3. **Evaluation**: Computes accuracy, classification report
4. **Visualization**: Generates confusion matrix heatmap
5. **Saving**: Exports model as `.pkl` file

## 🔧 Customization

### Use Your Own Data
Modify `train.py` to load your dataset:
```python
def load_your_data():
    # Replace this function
    df = pd.read_csv('your_data.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y
```

### Change the Model
Swap RandomForest for any scikit-learn model:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

### Adjust Training Schedule
Edit `.github/workflows/train.yml`:
```yaml
schedule:
  - cron: '0 */6 * * *'  # Run every 6 hours
```

## 📊 GitHub Actions Workflow

### Triggers
- **Push to main**: Trains on every commit
- **Daily Schedule**: Automatic training at 2 AM UTC
- **Manual**: Click "Run workflow" in Actions tab

### What It Does
1. 📋 Checks out code
2. 🐍 Sets up Python 3.10
3. 📦 Installs dependencies
4. 🔥 Runs training script
5. 📊 Uploads results as artifacts
6. 💾 Commits results back to repo

## 📈 Free Tier Limits

**GitHub Actions Free Tier:**
- 2,000 minutes/month for private repos
- **Unlimited** for public repos ✅
- This project uses ~2-3 minutes per training run
- You can train **600+ times/month** for free!

## 🎯 Use Cases

1. **Learning ML**: Perfect for understanding ML workflows
2. **Portfolio Projects**: Show automated MLOps skills
3. **Continuous Retraining**: Keep models updated with new data
4. **A/B Testing**: Compare different model architectures
5. **Experimentation**: Test hyperparameters automatically

## 🛠️ Technology Stack

- **Language**: Python 3.10
- **ML Library**: scikit-learn
- **CI/CD**: GitHub Actions
- **Visualization**: Matplotlib, Seaborn
- **Version Control**: Git/GitHub

## 📚 Learn More

- [GitHub Actions for ML](https://github.com/features/actions)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [MLOps Best Practices](https://ml-ops.org/)

## 👥 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📜 License

MIT License - Feel free to use this project for learning and commercial purposes!

## ⭐ Show Your Support

If this project helped you learn ML automation, give it a star ⭐!

---

**Built with 🔥 by @Infernodude777**
