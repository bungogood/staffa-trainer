# Staffa Trainer

Training for [staffa](https://github.com/bungogood/staffa) a backgammon engine

## Usage

```
git clone git@github.com:bungogood/staffa-trainer.git
cd staffa-trainer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Format

Data files should be csv files with the labels first and the input second e.g

```
win,wing,winbg,loseg,losebg,x_off,o_off,x_bar-1,x_bar-2,...
```

## References

-   [wildbg](https://github.com/carsten-wenderdel/wildbg) by [Carsten Wenderdel](https://github.com/carsten-wenderdel)
