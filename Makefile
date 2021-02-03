pytest: clean-capture
	PYTHONPATH='./src' python3 -m pytest tests

clean-capture:
	rm -f ./capture/*
