# Accessibility Path Finding
This project implements a multi-objective A* algorithm for accessibility path finding within the Claremont Colleges.  

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

# What are the folders?
1. `0-datasets` stores all of the datasets we have used in this project. Within it, `raw` stores the raw files, while `mst` stores all of the processed and clean files
2. `1-import` stores jupyter notebooks used to convert the data from their original formats to pandas dataframes.
3. `2-build` stores jupyter notebooks that build, or wrangle, all of the files from `raw` and output the processed files in the `mst` folder.
4. `3-analysis` stores the notebooks used for analysis and outputs graphics.
5. `4-export` stores the output (graphs) corrsponding to notebooks in `3-analysis`.
6. `pathfindingat5cs.py` store the main multi-objective A* algorithm, and several helper functions used by the main algorithm.
7. `utils.py` stores all other helper functions used to process files.  
8. We have included original links to several tutorials we borrowed from within the notebooks.

# Demostration
To see the actual results or start querying paths, run `01-mapping-astar-output.ipynb` within the `3-analysis` folder and change the `start` and `end` to the corresponding indices of actual locations. The outputs should be automatically generated within the notebook.

