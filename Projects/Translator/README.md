## This app will get audio to translate audio for you! And translate it back for you :)


### Prerequisites:
Let's set a **virtual environment** to be very safe
```bash
source venv/bin/activate
```
In the case you would want to leave the virtual environment, type:
```bash
deactivate
```
Just as a general note:

virtualenv avoids the need to install Python packages globally. When a virtualenv is active, pip will install packages within the environment, which does not affect the base Python installation in any way. [Stack Overflow](https://stackoverflow.com/questions/41972261/what-is-a-virtualenv-and-why-should-i-use-one)

## Install dependencies
pip install torch torchaudio transformers
