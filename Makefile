.PHONY: all install main

all: install main

install: requirements.txt
	pip install -r requirements.txt

main: main.py
	python main.py
