.PHONY: notebook chat requirements clean

notebook:
	python3 tools/make_colab.py

chat: requirements data/tokenizer.json checkpoints/
	python -m guppylm chat

data/tokenizer.json checkpoints/:
	python -m guppylm download

requirements: requirements.txt
	pip install -r requirements.txt

clean:
	- rm -rf venv data/tokenizer.json
