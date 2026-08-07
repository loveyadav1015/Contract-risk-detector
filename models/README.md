# Saved Models

This directory stores trained model checkpoints.

After training, the best checkpoint will be saved here by `src/train.py`.
The API service loads the model from this directory at startup.

**Do not commit large checkpoint files to version control.**
The `models/` directory contents (except this README) are listed in `.gitignore`.
