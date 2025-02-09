# Applications-of-AI
Repo will contain all the files necessary for the GBM which will include pre made projects as well as ones members will be able to follow along and fill in lines of code to realize their AI potential.

First for any project you need a virtual environment so you can **isolate** your Python dependencies
    - What if a project requires a specific version, but it's super old. You WOULDN'T want to change the version on your entire machine!


## How to run each project
#### Step 1
The first step will be to make a **virtual environment** so you can isolate the dependencies for your project to that environment. (Avoid changing your machine for a singular project)
```bash
python3 -m venv venv
```


#### Step 2
Now we will turn on the virtual environment
```bash
source venv/bin/activate
```

#### Step 4
Download dependencies. Using the requirements.txt file that was given to us, we run the following command:
```bash
pip install -r requirements.txt
```

#### Step 5
Now, it's time to actually run the project!
```python3
python3 demo.py
```
