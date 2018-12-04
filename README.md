# accessibility_path_finding
Accessibility path finding for the Claremont Colleges

# Setup
## Create global .gitignore file
This is so that your temp files will not get uploaded to the git repo. Follow these steps
1. Open terminal/command line
2. Execute `git config --global core.excludesfile ~/.gitignore_global` to add .gitignore_global as the global gitignore file in your global git config)
3. Execute `sudo nano ~/.gitignore_global` to edit your .gitignore_global file
4. Paste in the content from this link: https://www.gitignore.io/api/python
5. Use ctrl + x to exit and press y to save the file.

## Download dependencies:
1. Download and install Python 3 (if you don’t already have it) from here: https://www.python.org/downloads/  

2. Check that `pip`, a package manager of Python, is installed by going into the Terminal/Command Line and type `pip -V`. This will tell you what version of pip, if at all, is on your machine. You should see pip already installed if you are using Python downloaded from python.org. But if pip version does not show up, it means you don't yet have pip, so install it following this instruction: https://pip.pypa.io/en/stable/installing/  

3. Download Jupyter notebooks by executing in Terminal/Command line:

```
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```

Jupyter notebook is a tool that allows interactive coding in many languages, including Python. Our workshop will be done in this environment! See this guide if the above commands don't work for you: http://jupyter.org/install.  

4. Install Homebrew by following instructions in here: `https://brew.sh/`  

5. Install all the packages by execute in Terminal/Command line:  

```
pip install matplotlib pandas sklearn seaborn numpy osmnx
brew install spatialindex
```

## To run code

